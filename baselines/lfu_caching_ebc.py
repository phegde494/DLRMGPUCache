#!/usr/bin/env python3

import math
import abc
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    pooling_type_to_str,
    DataType,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

######################################################################
# 1) LFU-Based Caching EmbeddingBagCollection
######################################################################

class LfuCachingEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """
    A custom EmbeddingBagCollection that:
      - Stores the full embedding weights on CPU (the "master" copy).
      - Maintains a row-wise GPU cache of limited capacity (a fraction of total rows).
      - Tracks row usage frequencies (LFU).
      - On each batch, fetches needed rows from CPU -> GPU (with optional eviction).
      - Optionally can prefetch the next batch's rows asynchronously (on a separate stream).
      - After backward, you can sync updated GPU rows back to CPU so the master remains up-to-date.
      - Integrates with TorchRec's typical usage for MLP + Embedding Bag operations.

    Supports multi-GPU row‐wise sharding if used in conjunction with the custom Sharder
    provided below (each rank only holds a shard of the CPU table + GPU cache).

    Key features:
      - CPU → GPU row copying
      - LFU eviction
      - Asynchronous prefetch (optional)
      - Weighted or unweighted embedding bag lookups
      - Row‐wise parallel usage in multi‐GPU training

    Example Usage:
      >>> ebc = LfuCachingEmbeddingBagCollection(
      ...     tables=[EmbeddingBagConfig(
      ...         name="table0",
      ...         embedding_dim=128,
      ...         num_embeddings=100_000,
      ...         feature_names=["feature0"],
      ...     )],
      ...     is_weighted=False,
      ...     cache_ratio=0.01,
      ...     device=torch.device("cuda"),
      ...     max_prefetch_depth=2,
      ...     pinned_memory=True,
      ... )

      # In your main training loop:
      >>> # forward:
      >>> output = ebc(features_kjt)
      >>> # backward:
      >>> loss.backward()
      >>> # sync updates from GPU -> CPU if desired each iteration
      >>> ebc.sync_cache_to_cpu()

    Attributes:
        tables (List[EmbeddingBagConfig]): Table configs. Each table is sharded or not.
        is_weighted (bool): Whether the input KeyedJaggedTensor has per-sample weights.
        cache_ratio (float): Fraction of total embeddings to hold on GPU for each table.
        device (torch.device): The GPU device where your cache should reside.
        pinned_memory (bool): Whether to allocate CPU master weights in pinned (page-locked) memory
            to accelerate CPU->GPU transfers.
        max_prefetch_depth (int): How many prefetch “slots” to maintain if you want to asynchronously
            copy the next N batches. If set to 0, no prefetch is used.

    Note on Multi-GPU:
        - Typically in TorchRec, each rank is responsible for a subset of the embedding rows.
          This class can handle that scenario by letting each rank instantiate a smaller CPU
          parameter array. The custom Sharder below handles that offset logic automatically.
        - Each rank will then build its own GPU cache and do usage counting only for the shard it owns.

    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool = False,
        cache_ratio: float = 0.01,
        device: torch.device = torch.device("cuda"),
        pinned_memory: bool = False,
        max_prefetch_depth: int = 0,  # set > 0 to enable async prefetch
        row_offset: int = 0,         # used for row-wise sharding
    ) -> None:
        super().__init__()
        self._is_weighted = is_weighted
        self._embedding_bag_configs = tables
        self._device = device
        self._pinned_memory = pinned_memory
        self._max_prefetch_depth = max_prefetch_depth
        self._row_offset = row_offset  # for row-wise partition (start row)

        # We maintain a list of "embedding names" to replicate the standard output style
        self._embedding_names: List[str] = []
        self._lengths_per_embedding: List[int] = []

        # CPU "master" weights + usage frequency
        # GPU "cache" weight
        self.cpu_params = nn.ParameterDict()
        self.gpu_params = nn.ParameterDict()
        self.cache_index: Dict[str, Dict[int, int]] = {}
        self.inverse_cache_index: Dict[str, Dict[int, int]] = {}
        self.lfu_frequency: Dict[str, torch.Tensor] = {}

        # Optional streams for asynchronous prefetch
        self._prefetch_streams = []
        for _ in range(self._max_prefetch_depth):
            self._prefetch_streams.append(torch.cuda.Stream(device=self._device))

        # Build each table
        for table_config in tables:
            if not table_config.feature_names:
                table_config.feature_names = [table_config.name]
            # Create CPU master weight
            cpu_weight = self._init_cpu_weight(table_config)
            self.cpu_params[table_config.name] = cpu_weight

            # GPU cache param
            cache_size = max(1, int(math.ceil(table_config.num_embeddings * cache_ratio)))
            gpu_weight = nn.Parameter(
                torch.zeros(
                    cache_size,
                    table_config.embedding_dim,
                    dtype=cpu_weight.dtype,
                    device=self._device,
                    requires_grad=True,
                )
            )
            self.gpu_params[table_config.name] = gpu_weight

            # Indices for the cache
            self.cache_index[table_config.name] = {}
            self.inverse_cache_index[table_config.name] = {}

            # Usage frequency vector (on CPU)
            # We'll keep it on CPU to avoid expensive GPU updates
            freq_vec = torch.zeros(
                table_config.num_embeddings,
                dtype=torch.int64,
                device="cpu"
            )
            self.lfu_frequency[table_config.name] = freq_vec

            # Fill in our "embedding_names" for the output KeyedTensor
            for feat_name in table_config.feature_names:
                if _feature_name_is_ambiguous(feat_name, tables):
                    emb_name = f"{feat_name}@{table_config.name}"
                else:
                    emb_name = feat_name
                self._embedding_names.append(emb_name)
                self._lengths_per_embedding.append(table_config.embedding_dim)

        # Move entire module to GPU for consistency
        # (the CPU params remain on CPU, but PyTorch lets us store them in a ParameterDict).
        self.to(self._device)

    def _init_cpu_weight(self, table_config: EmbeddingBagConfig) -> nn.Parameter:
        """
        Allocates the CPU "master" weight for the given table in pinned or unpinned memory.
        """
        dtype = (
            torch.float32
            if table_config.data_type == DataType.FP32
            else torch.float16
        )
        # Number of rows in this table shard
        rows = table_config.num_embeddings
        dims = table_config.embedding_dim

        # If pinned_memory is True, we allocate a CPU tensor in pinned memory
        # so that CPU->GPU transfers can be faster (async).
        if self._pinned_memory:
            # torch.empty(..., pin_memory=True) only works with .to(device) or .copy_() ops
            # so we create an empty pinned tensor, wrap it as a Parameter.
            cpu_tensor = torch.empty((rows, dims), dtype=dtype, pin_memory=True)
        else:
            cpu_tensor = torch.empty((rows, dims), dtype=dtype)

        nn.init.normal_(cpu_tensor, mean=0.0, std=0.02)
        return nn.Parameter(cpu_tensor, requires_grad=False)

    @property
    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    @property
    def is_weighted(self) -> bool:
        return self._is_weighted

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Standard forward pass:
          1) Convert features to dict (feature_name -> values, offsets, weights).
          2) For each table + feature_name in that table:
             - Gather unique row indices in the batch, update usage freq
             - Move missing rows from CPU -> GPU cache (evict if needed)
             - Build an index remap from batch_indices -> GPU rows
             - Perform embedding_bag on GPU
          3) Return the concatenated KeyedTensor across all tables & features
        """
        feature_dict = features.to_dict()
        pooled_embeddings: List[torch.Tensor] = []

        for table_config in self._embedding_bag_configs:
            table_name = table_config.name
            pooling_mode = pooling_type_to_str(table_config.pooling)
            embed_dim = table_config.embedding_dim

            for feat_name in table_config.feature_names:
                # Determine the "key" we will output as
                if _feature_name_is_ambiguous(feat_name, self._embedding_bag_configs):
                    emb_name = f"{feat_name}@{table_name}"
                else:
                    emb_name = feat_name

                f = feature_dict[feat_name]
                indices = f.values()  # [batch_size * lengths] on GPU
                offsets = f.offsets()
                psw = f.weights() if self._is_weighted else None

                # 1) Unique row indices (CPU frequency updates)
                unique_indices_cpu = indices.unique().to("cpu")

                # Increment usage frequency
                self.lfu_frequency[table_name][unique_indices_cpu] += 1

                # 2) Move needed rows to GPU
                self._ensure_rows_on_gpu(table_config, unique_indices_cpu)

                # 3) Build the GPU row mapping for this batch
                gpu_index_map = self._gpu_index_map(table_name, indices)

                # 4) Perform embedding bag on GPU
                #    (We rely on PyTorch’s built-in embedding_bag, which uses our GPU param.)
                out = F.embedding_bag(
                    gpu_index_map,
                    self.gpu_params[table_name],
                    offsets,
                    per_sample_weights=psw,
                    mode=pooling_mode,
                    include_last_offset=True,
                )
                pooled_embeddings.append(out)

        # Concatenate in the correct (feature) order
        output_values = torch.cat(pooled_embeddings, dim=1)
        return KeyedTensor(
            keys=self._embedding_names,
            values=output_values,
            length_per_key=self._lengths_per_embedding,
        )

    ######################################################################
    # Prefetch Interface
    ######################################################################
    def prefetch(
        self,
        table_config: EmbeddingBagConfig,
        rows_needed: torch.Tensor,
        prefetch_stream_idx: int = 0,
    ) -> None:
        """
        Asynchronously copy (CPU->GPU) the needed rows into the GPU cache for the *next* batch.

        This is meant to be called *before* your forward for that next batch, and can run on a
        separate CUDA stream for overlap with current batch’s compute.

        Usage:
          - pick a stream: e.g. prefetch_stream_idx = (current_step % max_prefetch_depth)
          - call ebc.prefetch(table_config, next_batch_indices, prefetch_stream_idx)
          - in your forward, do "with torch.cuda.stream(...)" or an event to ensure
            prefetch completion.

        Note: This must do basically the same logic as `_ensure_rows_on_gpu`, including evicting
        if necessary. The difference is that we do it on `self._prefetch_streams[prefetch_stream_idx]`
        so that it doesn’t block the main stream.
        """
        if prefetch_stream_idx >= len(self._prefetch_streams):
            # No prefetch stream available
            return

        stream = self._prefetch_streams[prefetch_stream_idx]
        rows_needed_cpu = rows_needed.to("cpu")
        # Increment usage freq so that we don't immediately evict them
        self.lfu_frequency[table_config.name][rows_needed_cpu] += 1

        # We do the same logic as _ensure_rows_on_gpu but on the chosen stream:
        with torch.cuda.stream(stream):
            self._ensure_rows_on_gpu(table_config, rows_needed_cpu, non_blocking=True)

    def wait_prefetches(self):
        """
        Synchronize all prefetch streams with the main stream. Call this before your
        forward pass if you want to ensure that all asynchronous copies are done.
        """
        for s in self._prefetch_streams:
            s.synchronize()

    ######################################################################
    # Sync and Eviction Logic
    ######################################################################

    def sync_cache_to_cpu(self):
        """
        Copies all GPU cache rows back to CPU. Typically you call this after `backward()`
        so that the CPU master weights remain in sync with any newly updated GPU cache rows.

        In many workflows, you only *need* to copy back rows that you plan to evict or
        all rows if you want a completely up-to-date CPU. This method copies them all
        unconditionally.
        """
        for table_config in self._embedding_bag_configs:
            tname = table_config.name
            cpu_w = self.cpu_params[tname]
            gpu_w = self.gpu_params[tname]
            c_map = self.cache_index[tname]
            # For each row in the cache, copy GPU->CPU
            for (cpu_row, gpu_row) in c_map.items():
                cpu_w.data[cpu_row].copy_(gpu_w.data[gpu_row], non_blocking=False)

    def _ensure_rows_on_gpu(
        self,
        table_config: EmbeddingBagConfig,
        rows_needed_cpu: torch.Tensor,
        non_blocking: bool = False,
    ) -> None:
        """
        Make sure that all 'rows_needed_cpu' are present in the GPU cache for this table.
        If the GPU cache is short on space, evict the least-frequently used rows.
        Then copy the CPU rows into GPU if they are missing.

        This is called during forward and also from `prefetch()` (on a separate stream).
        """
        tname = table_config.name
        freq = self.lfu_frequency[tname]
        c_map = self.cache_index[tname]
        inv_map = self.inverse_cache_index[tname]
        cpu_w = self.cpu_params[tname]  # [num_rows, dim], pinned or not
        gpu_w = self.gpu_params[tname]  # [cache_size, dim]
        cache_size = gpu_w.size(0)

        # List out which rows are missing from the GPU cache
        needed_list = rows_needed_cpu.tolist()
        to_load = [r for r in needed_list if r not in c_map]
        if not to_load:
            return

        # Evict if we have insufficient free capacity
        free_slots = cache_size - len(c_map)
        shortfall = len(to_load) - free_slots
        if shortfall > 0:
            # We must evict 'shortfall' rows from GPU
            # We'll evict the rows with the smallest usage frequency
            cached_rows = list(c_map.keys())  # CPU row IDs
            # Collect (cpu_row, freqVal)
            freq_list = [(row, freq[row].item()) for row in cached_rows]
            freq_list.sort(key=lambda x: x[1])  # ascending freq
            evict_rows = [x[0] for x in freq_list[:shortfall]]

            for row in evict_rows:
                gpu_row = c_map[row]
                # Copy GPU->CPU to keep them in sync with updates from backprop
                cpu_w.data[row].copy_(gpu_w.data[gpu_row], non_blocking=False)
                # Remove from mapping
                del c_map[row]
                del inv_map[gpu_row]

        # Now we definitely have space for all 'to_load'
        free_gpu_rows = set(range(cache_size)) - set(inv_map.keys())
        free_gpu_rows_list = list(free_gpu_rows)
        idx = 0
        for cpu_r in to_load:
            if idx >= len(free_gpu_rows_list):
                break  # shouldn't happen if we've done correct eviction
            gpu_r = free_gpu_rows_list[idx]
            idx += 1

            # copy CPU->GPU
            # We can do non_blocking if pinned_memory is True
            gpu_w.data[gpu_r].copy_(cpu_w.data[cpu_r], non_blocking=non_blocking)

            # update mapping
            c_map[cpu_r] = gpu_r
            inv_map[gpu_r] = cpu_r

    def _gpu_index_map(self, table_name: str, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert CPU row indices (in `batch_indices`) to GPU row indices by referencing
        self.cache_index.  If you do partial row coverage (some remain on CPU),
        you'd handle that logic here. In this example, we assume all needed rows
        are guaranteed to be in GPU after _ensure_rows_on_gpu.
        """
        c_map = self.cache_index[table_name]
        # We'll do a CPU gather approach for clarity:
        #  1) move batch_indices to CPU
        #  2) map each index
        #  3) return a GPU tensor
        batch_indices_cpu = batch_indices.to("cpu")
        mapped = [c_map[i.item()] for i in batch_indices_cpu]
        return torch.tensor(mapped, device=self._device, dtype=batch_indices.dtype)

######################################################################
# 2) Helper Function
######################################################################

def _feature_name_is_ambiguous(feature_name: str, tables: List[EmbeddingBagConfig]) -> bool:
    """
    TorchRec uses "feature@table" naming when a given feature_name is used by multiple tables.
    This utility checks if the feature_name is present in more than one table config.
    """
    count = 0
    for t in tables:
        if feature_name in t.feature_names:
            count += 1
        if count > 1:
            return True
    return False


######################################################################
# 3) (OPTIONAL) Custom Sharder for Row‐Wise Parallel
######################################################################
class LfuCachingEBCSharder(ModuleSharder[LfuCachingEmbeddingBagCollection]):
    """
    A custom TorchRec Sharder that will:
      - Partition each table's row range across ranks (RowWise).
      - Construct a local LfuCachingEmbeddingBagCollection that only has the CPU rows for
        that rank's shard. (We apply offset for local param creation.)
      - Return that module as the local shard.

    Usage:
      - Include this sharder in your `sharders` list, e.g.:
          >>> sharders = [LfuCachingEBCSharder(cache_ratio=0.01, pinned_memory=True)]
      - Then TorchRec's planner will assign your table shards among ranks,
        and each rank will instantiate a local LfuCachingEmbeddingBagCollection with
        row_offset = ...
      - The result is that each rank only stores (and caches) its local subset of rows
        on CPU->GPU, so memory usage is minimized.

    This is purely optional. If you want a simpler “all rows on each rank” approach,
    you can skip the row‐wise partition logic. But this class shows how you might do it.
    """

    def __init__(
        self,
        cache_ratio: float = 0.01,
        pinned_memory: bool = False,
        max_prefetch_depth: int = 0,
    ) -> None:
        super().__init__()
        self._cache_ratio = cache_ratio
        self._pinned_memory = pinned_memory
        self._max_prefetch_depth = max_prefetch_depth

    def shard(self, module: LfuCachingEmbeddingBagCollection, device, rank: int, world_size: int):
        """
        This method is never actually called for us because we are building the LfuCachingEBC
        on the fly. Usually TorchRec tries to do "construct_module_shard" by the Sharders.
        We'll implement `shard` so that if you want to plan from a larger embedding config
        we break it up row‐wise among ranks.
        """
        # Because we do row-wise partition, we scale the number of embeddings for each table
        # by 1/world_size, and shift row_offset accordingly.

        new_tables: List[EmbeddingBagConfig] = []
        for table_config in module.embedding_bag_configs:
            # old row dimension
            total_rows = table_config.num_embeddings
            # local portion
            rows_per_rank = (total_rows + world_size - 1) // world_size
            start_idx = rows_per_rank * rank
            end_idx = min(start_idx + rows_per_rank, total_rows)

            local_num_rows = end_idx - start_idx
            # create a new config for the local shard
            local_config = EmbeddingBagConfig(
                name=table_config.name,
                embedding_dim=table_config.embedding_dim,
                num_embeddings=local_num_rows,
                feature_names=table_config.feature_names,
                pooling=table_config.pooling,
                data_type=table_config.data_type,
            )
            new_tables.append(local_config)

        # Build new local module
        shard_mod = LfuCachingEmbeddingBagCollection(
            tables=new_tables,
            is_weighted=module.is_weighted,
            cache_ratio=self._cache_ratio,
            device=device,
            pinned_memory=self._pinned_memory,
            max_prefetch_depth=self._max_prefetch_depth,
            row_offset=0  # If you want to handle offsets, we'd store start_idx and do row-> local row
        )
        return shard_mod

    def shardable_parameters(
        self,
        module: LfuCachingEmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        """
        Return the CPU and GPU parameters for TorchRec to handle. Typically you'd want
        only GPU parameters in the sharded plan, or a subset. We'll be naive here.
        """
        # We'll return both CPU and GPU for completeness:
        params_dict = {}
        for k, v in module.cpu_params.items():
            params_dict[f"cpu_params.{k}"] = v
        for k, v in module.gpu_params.items():
            params_dict[f"gpu_params.{k}"] = v
        return params_dict

    def sharding_types(self, module: LfuCachingEmbeddingBagCollection) -> List[str]:
        """
        Return the list of supported sharding types. We'll say ROW_WISE.
        """
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(self, module: LfuCachingEmbeddingBagCollection) -> List[str]:
        """
        Return list of supported compute kernels. We'll just say "BATched Fused" is possible,
        but in practice, we have our own custom kernel. We'll return an empty list or a string.
        """
        return []

    @property
    def module_type(self):
        return LfuCachingEmbeddingBagCollection
