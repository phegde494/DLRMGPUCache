#!/usr/bin/env python3
import copy
import os
import math
from typing import Dict, List, Optional, Union, cast, Any, Callable, Set, Tuple

import torch
import torch.distributed as dist
from functools import reduce
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.perf_models import NoopPerfModel  # Note: lowercase 'o' in Torchrec 0.1.1
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner
from torchrec.distributed.planner.proposers import GreedyProposer, UniformProposer
from torchrec.distributed.planner.stats import EmbeddingStats  # TorchRec 0.1.1 uses this instead of NoOpStats
from torchrec.distributed.planner.storage_reservations import HeuristicalStorageReservation
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    Enumerator,
    StorageReservation,
    Proposer,
    Partitioner,
    PerfModel,
    Stats,
    Topology,
    ShardingOption,
    Shard,
    Storage,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingType,
    ShardingPlan,
    EnumerableShardingSpec,
    ShardMetadata,
    ParameterSharding,
)
from torch import nn

def get_local_size() -> int:
    """
    Get local world size (number of GPUs on current node)
    """
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


class CPUGPUMixedShardingPlanner(EmbeddingShardingPlanner):
    """
    Sharding planner that places a portion of embedding tables on CPU memory
    to reduce GPU memory usage while maintaining most performance benefits.
    
    Args:
        cpu_offload_ratio: Fraction of embedding tables to place on CPU (0.0-1.0)
        topology: Device topology information
        batch_size: Training batch size
        constraints: Sharding constraints for parameters
        debug: Enable debug logging
    """

    def __init__(
        self,
        cpu_offload_ratio: float = 0.3,
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = False,
    ) -> None:
        # Create default topology if not provided
        if topology is None:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            topology = Topology(
                world_size=dist.get_world_size() if dist.is_initialized() else 1,
                compute_device=default_device,
                local_world_size=get_local_size(),
            )
        
        # Create default components for the planner
        enumerator = EmbeddingEnumerator(topology=topology, batch_size=batch_size)
        storage_reservation = HeuristicalStorageReservation(percentage=0.15)
        proposer = [GreedyProposer(), UniformProposer()] 
        partitioner = GreedyPerfPartitioner()
        perf_model = NoopPerfModel(topology=topology)  # Note the lowercase 'o'
        stats = EmbeddingStats(topology=topology)  # TorchRec 0.1.1 uses this
        
        # Initialize parent class with all components
        super().__init__(
            topology=topology,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            proposer=proposer,
            partitioner=partitioner,
            perf_model=perf_model,  # Note 0.1.1 uses perf_model not performance_model
            stats=stats,
            constraints=constraints,
        )
        
        # Store CPU offload ratio (clamp between 0 and 1)
        self._cpu_offload_ratio = max(0.0, min(1.0, cpu_offload_ratio))
        self._debug = debug
        # Store batch size
        self._batch_size = batch_size
        
        print(f"Initialized CPUGPUMixedShardingPlanner with cpu_offload_ratio={self._cpu_offload_ratio}")

    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """
        Generate a sharding plan for the given module with CPU offloading.
        
        Args:
            module: Module to be sharded
            sharders: List of module sharders 
        
        Returns:
            ShardingPlan with mixed CPU/GPU placements
        """
        # Get the standard plan first
        standard_plan = super().plan(module, sharders)
        
        # If CPU offloading is enabled, modify the plan
        if self._cpu_offload_ratio > 0:
            return self._modify_plan_for_cpu_offload(standard_plan)
        
        return standard_plan
    
    def _modify_plan_for_cpu_offload(self, original_plan: ShardingPlan) -> ShardingPlan:
        """
        Modifies the plan to place a portion of tables on CPU
        """
        if self._debug:
            print(f"Modifying plan for CPU offloading with ratio: {self._cpu_offload_ratio}")
        
        # Create a deep copy of the plan to modify
        modified_plan = copy.deepcopy(original_plan)
        
        for module_path, module_plan in modified_plan.plan.items():
            # Get parameters sorted by size (approximately - just using number of shards)
            param_info = []
            for param_name, param_sharding in module_plan.items():
                if param_sharding.sharding_spec is not None and isinstance(
                    param_sharding.sharding_spec, EnumerableShardingSpec
                ):
                    param_info.append((param_name, len(param_sharding.sharding_spec.shards)))
            
            # Sort parameters by number of shards (larger first)
            param_info.sort(key=lambda x: x[1], reverse=True)
            
            # Determine how many parameters to move to CPU
            num_params = len(param_info)
            num_cpu_params = max(1, int(num_params * self._cpu_offload_ratio))
            cpu_params = set(name for name, _ in param_info[:num_cpu_params])
            
            # Modify each parameter according to CPU/GPU assignment
            for param_name, param_sharding in module_plan.items():
                # Skip parameters without enumerable spec
                if param_sharding.sharding_spec is None or not isinstance(
                    param_sharding.sharding_spec, EnumerableShardingSpec
                ):
                    continue
                
                # Check if this parameter should be on CPU
                if param_name in cpu_params:
                    if self._debug:
                        print(f"Parameter {param_name}: moving to CPU")
                    
                    # Move all shards to CPU
                    new_shards = []
                    for shard in param_sharding.sharding_spec.shards:
                        # Parse the rank from the original placement
                        current_placement = shard.placement
                        rank = int(current_placement.split(":")[1].split("/")[0])
                        
                        # Create new placement string for CPU
                        new_placement = f"rank:{rank}/cpu"
                        
                        # Create a new ShardMetadata with the CPU placement
                        new_shard = ShardMetadata(
                            shard_sizes=shard.shard_sizes,
                            shard_offsets=shard.shard_offsets,
                            placement=new_placement,
                        )
                        new_shards.append(new_shard)
                    
                    # Update the sharding spec with new shards
                    param_sharding.sharding_spec = EnumerableShardingSpec(new_shards)
        
        return modified_plan