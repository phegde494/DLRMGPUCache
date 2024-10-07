# The infrastructures of DLRM are mainly inspired by TorchRec:
# https://github.com/pytorch/torchrec/blob/main/torchrec/models/dlrm.py
import os
from contextlib import nullcontext
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import record_function

from baselines.models.dlrm import DenseArch, OverArch, InteractionArch, choose
from ..utils import get_time_elapsed
from ..datasets.utils import KJTAllToAll

import colossalai
from colossalai.nn.parallel.layers import ParallelFreqAwareEmbeddingBag, EvictionStrategy
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode

dist_logger = colossalai.logging.get_dist_logger()


def sparse_embedding_shape_hook(embeddings, feature_size, batch_size):
    return embeddings.view(feature_size, batch_size, -1).transpose(0, 1)


class FusedSparseModules(nn.Module):

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 fused_op='all_to_all',
                 reduction_mode='sum',
                 sparse=False,
                 output_device_type=None,
                 use_cache=False,
                 cache_sets=500_000,
                 cache_lines=1,
                 id_freq_map=None,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 is_dist_dataloader=True,
                 use_lfu_eviction=False):
        super(FusedSparseModules, self).__init__()
        if use_cache:
            self.embed = ParallelFreqAwareEmbeddingBag(
                sum(num_embeddings_per_feature),
                embedding_dim,
                sparse=sparse,
                mode=reduction_mode,
                include_last_offset=True,
                cuda_row_num=cache_sets,
                ids_freq_mapping=id_freq_map,
                warmup_ratio=warmup_ratio,
                buffer_size=buffer_size,
                evict_strategy=EvictionStrategy.LFU if use_lfu_eviction else EvictionStrategy.DATASET
            )
        else:
            # raise NotImplementedError("Other EmbeddingBags are under development")
            self.embed = nn.EmbeddingBag(
                sum(num_embeddings_per_feature),
                embedding_dim,
                sparse=sparse,
                mode=reduction_mode,
                include_last_offset=True
            )

        if is_dist_dataloader:
            self.kjt_collector = KJTAllToAll(gpc.get_group(ParallelMode.GLOBAL))
        else:
            self.kjt_collector = None

    def forward(self, sparse_features):
        if self.kjt_collector:
            with record_function("(zhg)KJT AllToAll collective"):
                sparse_features = self.kjt_collector.all_to_all(sparse_features)

        keys, batch_size = sparse_features.keys(), sparse_features.stride()

        # flattened_sparse_embeddings = self.embed(
        #     sparse_features.values(),
        #     sparse_features.offsets(),
        #     shape_hook=lambda x: sparse_embedding_shape_hook(x, len(keys), batch_size))

        if hasattr(self.embed, 'forward') and 'shape_hook' in self.embed.forward.__code__.co_varnames:
            # Case with cache enabled (e.g., using ParallelFreqAwareEmbeddingBag)
            flattened_sparse_embeddings = self.embed(
                sparse_features.values(),
                sparse_features.offsets(),
                shape_hook=lambda x: sparse_embedding_shape_hook(x, len(keys), batch_size)
            )
        else:
            # Fallback case: standard EmbeddingBag usage without shape_hook
            flattened_sparse_embeddings = self.embed(
                sparse_features.values(),
                sparse_features.offsets()
            )
        # print("EMBEDDING SHAPE = ", flattened_sparse_embeddings.shape)
        # if flattened_sparse_embeddings.dim() == 2:
        #     flattened_sparse_embeddings = flattened_sparse_embeddings.unsqueeze(1)
        flattened_sparse_embeddings = flattened_sparse_embeddings.view(16384, 13, 128)
        # print("EMBEDDING SHAPE = ", flattened_sparse_embeddings.shape)
        return flattened_sparse_embeddings


class FusedDenseModules(nn.Module):
    """
    Fusing dense operations of DLRM into a single module
    """

    def __init__(self, embedding_dim, num_sparse_features, dense_in_features, dense_arch_layer_sizes,
                 over_arch_layer_sizes):
        super(FusedDenseModules, self).__init__()
        if dense_in_features <= 0:
            self.dense_arch = nn.Identity()
            over_in_features = choose(num_sparse_features, 2)
            num_dense = 0
        else:
            self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes)
            over_in_features = (embedding_dim + choose(num_sparse_features, 2) + num_sparse_features)
            num_dense = 1

        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features, num_dense_features=num_dense)
        self.over_arch = OverArch(in_features=over_in_features, layer_sizes=over_arch_layer_sizes)

    def forward(self, dense_features, embedded_sparse_features):
        embedded_dense_features = self.dense_arch(dense_features)
        concat_dense = self.inter_arch(dense_features=embedded_dense_features, sparse_features=embedded_sparse_features)
        logits = self.over_arch(concat_dense)

        return logits


class HybridParallelDLRM(nn.Module):
    """
    Model parallelized Embedding followed by Data parallelized dense modules
    """

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 num_sparse_features,
                 dense_in_features,
                 dense_arch_layer_sizes,
                 over_arch_layer_sizes,
                 dense_device,
                 sparse_device,
                 sparse=False,
                 fused_op='all_to_all',
                 use_cache=False,
                 cache_sets=500_000,
                 cache_lines=1,
                 id_freq_map=None,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 is_dist_dataloader=True,
                 use_lfu_eviction=False):

        super(HybridParallelDLRM, self).__init__()
        if use_cache and sparse_device.type != dense_device.type:
            raise ValueError(f"Sparse device must be the same as dense device, "
                             f"however we got {sparse_device.type} for sparse, {dense_device.type} for dense")

        self.dense_device = dense_device
        self.sparse_device = sparse_device

        self.sparse_modules = FusedSparseModules(num_embeddings_per_feature,
                                                 embedding_dim,
                                                 fused_op=fused_op,
                                                 sparse=sparse,
                                                 output_device_type=dense_device.type,
                                                 use_cache=use_cache,
                                                 cache_sets=cache_sets,
                                                 cache_lines=cache_lines,
                                                 id_freq_map=id_freq_map,
                                                 warmup_ratio=warmup_ratio,
                                                 buffer_size=buffer_size,
                                                 is_dist_dataloader=is_dist_dataloader,
                                                 use_lfu_eviction=use_lfu_eviction).to(sparse_device)
        self.dense_modules = DDP(module=FusedDenseModules(embedding_dim, num_sparse_features, dense_in_features,
                                                          dense_arch_layer_sizes,
                                                          over_arch_layer_sizes).to(dense_device),
                                 device_ids=[0 if os.environ.get("NVT_TAG", None) else gpc.get_global_rank()],
                                 process_group=gpc.get_group(ParallelMode.GLOBAL),
                                 gradient_as_bucket_view=True,
                                 broadcast_buffers=False,
                                 static_graph=True)

        # precompute for parallelized embedding
        param_amount = sum(num_embeddings_per_feature) * embedding_dim
        param_storage = self.sparse_modules.embed.weight.element_size() * param_amount
        param_amount += sum(p.numel() for p in self.dense_modules.parameters())
        param_storage += sum(p.numel() * p.element_size() for p in self.dense_modules.parameters())

        buffer_amount = sum(b.numel() for b in self.sparse_modules.buffers()) + \
            sum(b.numel() for b in self.dense_modules.buffers())
        buffer_storage = sum(b.numel() * b.element_size() for b in self.sparse_modules.buffers()) + \
            sum(b.numel() * b.element_size() for b in self.dense_modules.buffers())
        stat_str = f"Number of model parameters: {param_amount:,}, storage overhead: {param_storage/1024**3:.2f} GB. " \
                   f"Number of model buffers: {buffer_amount:,}, storage overhead: {buffer_storage/1024**3:.2f} GB."
        self.stat_str = stat_str

    def forward(self, dense_features, sparse_features, inspect_time=False):
        ctx1 = get_time_elapsed(dist_logger, "embedding lookup in forward pass") \
            if inspect_time else nullcontext()
        with ctx1:
            with record_function("Embedding lookup:"):
                # B // world size, sparse feature dim, embedding dim
                embedded_sparse = self.sparse_modules(sparse_features)

        ctx2 = get_time_elapsed(dist_logger, "dense operations in forward pass") \
            if inspect_time else nullcontext()
        with ctx2:
            with record_function("Dense operations:"):
                # B // world size, 1
                logits = self.dense_modules(dense_features, embedded_sparse)

        return logits

    def model_stats(self, prefix=""):
        return f"{prefix}: {self.stat_str}"
