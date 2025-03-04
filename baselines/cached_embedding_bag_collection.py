#!/usr/bin/env python3
"""
Drop–in replacement for TorchRec’s EmbeddingBagCollection that keeps the master
embedding tables on CPU and maintains a small GPU cache (with LRU eviction) of
frequently accessed rows. A custom autograd function is used so that gradients
computed on the GPU cached copies are properly “scattered” back into the CPU master.

Integration instructions:
  1. Save this file (e.g. as cached_embedding_bag_collection.py) in your project.
  2. In your dlrm_main.py (and dlrm.py if needed), change the import:
         from torchrec import EmbeddingBagCollection
     to:
         from cached_embedding_bag_collection import CachedEmbeddingBagCollection as EmbeddingBagCollection
  3. In your model initialization (in dlrm_main.py), change the call from:
         EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))
     to something like:
         EmbeddingBagCollection(tables=eb_configs, cache_ratio=0.01, device=torch.device("meta"))
     (You can adjust cache_ratio as desired.)
  4. All command–line arguments and training scripts remain unchanged.
  
NOTE:  
  • This implementation assumes pooling with include_last_offset=True and mode in {"sum", "mean"}.  
  • Distributed sharding is not supported in this drop–in (each rank gets its own full copy of each table).
  • If you see high GPU memory usage, try reducing the cache_ratio.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from collections import OrderedDict
from typing import List, Optional
import threading

# Import TorchRec types – adjust these imports if needed.
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig, pooling_type_to_str, DataType

################################################################################
# Custom Autograd Function: CachedEmbeddingBagFunction
################################################################################

class CachedEmbeddingBagFunction(Function):
    @staticmethod
    def forward(ctx, cpu_weight, input, offsets, mode, include_last_offset, cache_obj):
        """
        In forward, we delegate to cache_obj._forward_with_cache(). We save input
        and offsets on CPU for the backward.
        """
        input_cpu = input.cpu().long()
        offsets_cpu = offsets.cpu().long()
        out = cache_obj._forward_with_cache(input_cpu, offsets_cpu)
        ctx.save_for_backward(input_cpu, offsets_cpu)
        ctx.mode = mode
        ctx.num_embeddings = cpu_weight.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In backward, we “scatter–add” the gradient from each bag to the CPU master parameter.
        This ensures that even though the forward used the GPU cache, gradients update cpu_weight.
        """
        input_cpu, offsets_cpu = ctx.saved_tensors  # both on CPU
        mode = ctx.mode
        num_embeddings = ctx.num_embeddings
        embedding_dim = grad_output.shape[1]
        lengths = offsets_cpu[1:] - offsets_cpu[:-1]  # shape: [B]
        B = lengths.shape[0]
        bag_ids = torch.repeat_interleave(torch.arange(B, device=input_cpu.device), lengths)
        grad_output_cpu = grad_output.cpu()
        grad_per_element = grad_output_cpu[bag_ids]  # (N, embedding_dim)
        if mode == "mean":
            expanded_lengths = torch.repeat_interleave(lengths.float().unsqueeze(1), lengths, dim=0)
            grad_per_element = grad_per_element / expanded_lengths.clamp(min=1)
        grad_cpu = torch.zeros(num_embeddings, embedding_dim, device=input_cpu.device)
        grad_cpu = grad_cpu.index_add(0, input_cpu, grad_per_element)
        # Only cpu_weight is trainable.
        return grad_cpu, None, None, None, None, None

################################################################################
# CachedEmbeddingBag: A Drop–in Replacement for a Single Embedding Table
################################################################################

class CachedEmbeddingBag(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str,
        cache_ratio: float,
        include_last_offset: bool,
        dtype: torch.dtype,
        cache_device: torch.device,
    ):
        """
        Args:
            num_embeddings (int): total rows.
            embedding_dim (int): embedding dimension.
            mode (str): pooling mode ("sum" or "mean").
            cache_ratio (float): fraction of rows to keep on GPU.
            include_last_offset (bool): only True is supported.
            dtype (torch.dtype): data type.
            cache_device (torch.device): device to hold the cache (typically a GPU).
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.include_last_offset = include_last_offset

        # Master embedding table on CPU.
        self.cpu_weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=torch.device("cpu"))
        )
        nn.init.normal_(self.cpu_weight, mean=0, std=0.01)

        # Allocate the GPU cache.
        self.cache_size = max(1, int(num_embeddings * cache_ratio))
        self.cache_device = cache_device
        self.register_buffer(
            "cache_weight",
            torch.empty(self.cache_size, embedding_dim, dtype=dtype, device=self.cache_device)
        )
        # Use an OrderedDict for efficient LRU management.
        self.lru = OrderedDict()  # keys: global index, values: cache slot (an int)
        self._lock = threading.Lock()

        # Preload the cache with the first cache_size rows.
        for i in range(self.cache_size):
            self.cache_weight[i].copy_(self.cpu_weight[i].to(self.cache_device))
            self.lru[i] = i

    def _forward_with_cache(self, input_cpu: torch.Tensor, offsets_cpu: torch.Tensor) -> torch.Tensor:
        """
        This method gathers the embeddings for the given input (a 1D tensor of indices,
        on CPU) using the GPU cache. For any index missing in the cache, its embedding is
        fetched from CPU and (if possible) inserted into the cache—without evicting any index
        that is needed for the current forward pass.
        """
        unique_indices, inverse_indices = torch.unique(input_cpu, return_inverse=True)
        unique_indices_list = unique_indices.tolist()
        local_cache = {}  # mapping from index to tensor (on cache_device)

        with self._lock:
            for idx in unique_indices_list:
                if idx in self.lru:
                    # Cache hit: record the cache slot and update LRU order.
                    local_cache[idx] = self.cache_weight[self.lru[idx]]
                    self.lru.move_to_end(idx)
                else:
                    # Cache miss: fetch from CPU.
                    emb = self.cpu_weight[idx].to(self.cache_device)
                    local_cache[idx] = emb
                    # Insert into cache if possible.
                    if len(self.lru) < self.cache_size:
                        slot = len(self.lru)
                        self.lru[idx] = slot
                        self.cache_weight[slot].copy_(emb)
                    else:
                        # Try to evict a key not needed in this forward pass.
                        evict_key = None
                        for key in list(self.lru.keys()):
                            if key not in unique_indices_list:
                                evict_key = key
                                break
                        if evict_key is not None:
                            _, slot = self.lru.popitem(last=False)
                            self.lru[idx] = slot
                            self.cache_weight[slot].copy_(emb)
                        # Otherwise, do not update the cache (the current forward will use the local copy).
        # Build tensor for unique embeddings in the order of unique_indices_list.
        unique_emb_tensor = torch.stack([local_cache[idx] for idx in unique_indices_list], dim=0)
        # Reconstruct the per–lookup tensor using inverse_indices.
        gathered = unique_emb_tensor[inverse_indices.to(self.cache_device)]
        # Now perform bag pooling.
        if self.include_last_offset:
            num_bags = offsets_cpu.shape[0] - 1
            offsets_gpu = offsets_cpu.to(self.cache_device)
            lengths = offsets_gpu[1:] - offsets_gpu[:-1]
            bag_ids = torch.repeat_interleave(torch.arange(num_bags, device=self.cache_device), lengths)
            bag_embeddings = torch.zeros((num_bags, self.embedding_dim), device=self.cache_device, dtype=gathered.dtype)
            bag_embeddings = bag_embeddings.index_add(0, bag_ids, gathered)
            if self.mode == "mean":
                lengths = lengths.to(bag_embeddings.device).unsqueeze(1).clamp(min=1)
                bag_embeddings = bag_embeddings / lengths
            return bag_embeddings
        else:
            raise NotImplementedError("Non-include_last_offset pooling is not implemented.")

    def forward(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if per_sample_weights is not None:
            raise NotImplementedError("Weighted forward not implemented in CachedEmbeddingBag.")
        return CachedEmbeddingBagFunction.apply(self.cpu_weight, input, offsets, self.mode, self.include_last_offset, self)

    def sync_cache(self):
        """
        (Optional) Synchronize the GPU cache with updated CPU weights.
        Call this (for example) after an optimizer step.
        """
        with self._lock:
            for idx, slot in self.lru.items():
                self.cache_weight[slot].copy_(self.cpu_weight[idx].to(self.cache_device))

################################################################################
# CachedEmbeddingBagCollection: Drop–in Replacement for EmbeddingBagCollection
################################################################################

class CachedEmbeddingBagCollection(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        cache_ratio: float = 0.01,
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            tables (List[EmbeddingBagConfig]): list of table configurations.
            cache_ratio (float): fraction of each table to keep on GPU.
            is_weighted (bool): whether per–sample weights are used.
            device (torch.device, optional): device for the GPU cache.
                If device is None or meta, defaults to cuda:0 if available.
        """
        super().__init__()
        self._is_weighted = is_weighted
        self._embedding_bag_configs = tables  # store the table configs

        # Build shared–feature bookkeeping (to disambiguate duplicate feature names).
        shared_feature = {}
        for config in tables:
            for feature_name in config.feature_names:
                if feature_name in shared_feature:
                    shared_feature[feature_name] = True
                else:
                    shared_feature[feature_name] = False
        embedding_names = []
        for config in tables:
            for feature_name in config.feature_names:
                if shared_feature[feature_name]:
                    embedding_names.append(f"{feature_name}@{config.name}")
                else:
                    embedding_names.append(feature_name)
        self._embedding_names = embedding_names

        # Determine the cache device. If device is None or meta, default to cuda:0 if available.
        if device is None or device.type == "meta":
            self.cache_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.cache_device = device

        # Create one CachedEmbeddingBag per table.
        self.embedding_bags = nn.ModuleDict()
        for config in tables:
            # Use the proper dtype (compare with TorchRec’s DataType enum).
            dtype = torch.float32 if config.data_type == DataType.FP32 else torch.float16
            self.embedding_bags[config.name] = CachedEmbeddingBag(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.embedding_dim,
                mode=pooling_type_to_str(config.pooling),
                cache_ratio=cache_ratio,
                include_last_offset=True,
                dtype=dtype,
                cache_device=self.cache_device
            )

    @property
    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    @property
    def is_weighted(self) -> bool:
        return self._is_weighted

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): a sparse input.
        Returns:
            KeyedTensor: concatenated pooled embeddings.
        """
        pooled_embeddings = []
        feature_dict = features.to_dict()
        # For each table, run its CachedEmbeddingBag on each feature.
        for config in self._embedding_bag_configs:
            cached_eb = self.embedding_bags[config.name]
            for feature_name in config.feature_names:
                f = feature_dict[feature_name]
                pooled = cached_eb(
                    input=f.values(),
                    offsets=f.offsets(),
                    per_sample_weights=f.weights() if self._is_weighted else None
                )
                pooled_embeddings.append(pooled)
        data = torch.cat(pooled_embeddings, dim=1)
        # Return a KeyedTensor with disambiguated keys.
        return KeyedTensor(
            keys=self._embedding_names,
            values=data,
            length_per_key=[config.embedding_dim for config in self._embedding_bag_configs for _ in config.feature_names],
        )
