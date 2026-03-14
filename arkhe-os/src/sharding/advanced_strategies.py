# arkhe-os/src/sharding/advanced_strategies.py
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import heapq

PHI = 1.618033988749895

@dataclass
class ShardMetadata:
    """Metadata for a model shard."""
    shard_id: int
    layer_range: Tuple[int, int]
    parameter_count: int
    memory_bytes: int
    compute_flops: int
    coherence_requirement: float
    frequency_band: str  # 'bt7', 'wifi_pi', 'arkhe_c', 'thz'

class AdvancedShardingEngine:
    """
    Advanced model sharding strategies for distributed inference.
    """

    def __init__(self, model_config: Dict, cluster_info: Dict):
        self.model_config = model_config
        self.cluster = cluster_info
        self.shard_map = {}

    def allocate_coherence_aware(self, shards: List[ShardMetadata], nodes: List[Dict]) -> Dict[str, List[ShardMetadata]]:
        """
        Allocate shards based on node coherence and compute capability.
        """
        # Sort nodes by coherence (descending)
        sorted_nodes = sorted(nodes, key=lambda n: n['coherence'], reverse=True)
        # Sort shards by coherence requirement (descending)
        sorted_shards = sorted(shards, key=lambda s: s.coherence_requirement, reverse=True)

        allocation = {n['id']: [] for n in nodes}
        remaining_capacity = {n['id']: n['memory_available'] for n in nodes}

        for shard in sorted_shards:
            best_node = None
            best_score = -float('inf')

            for node in sorted_nodes:
                if node['coherence'] < shard.coherence_requirement:
                    continue
                if remaining_capacity[node['id']] < shard.memory_bytes:
                    continue

                coherence_margin = node['coherence'] - shard.coherence_requirement
                memory_util = 1.0 - remaining_capacity[node['id']] / node['memory_available']
                score = coherence_margin * 10 + (1 - memory_util)

                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                print(f"Warning: No suitable node for shard {shard.shard_id}")
                continue

            allocation[best_node['id']].append(shard)
            remaining_capacity[best_node['id']] -= shard.memory_bytes

        return allocation
