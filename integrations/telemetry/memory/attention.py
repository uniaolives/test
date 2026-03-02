# attention.py
# Memory ID 37-A: Temporal Attention with Cross-Domain Bridge Preservation
# Final Pre-T+0 Validation Layer

import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
import asyncio
import logging
from enum import Enum
import time

logger = logging.getLogger("MAT-Shadow")

class MemoryPriority(Enum):
    """Priority classes for memory retention"""
    ANCHOR_MASTER = 4      # KARNAK-sealed, high Φ, high centrality
    ETHICAL_ANCHOR = 3     # β > 0.75, used for SASC decisions
    RECENT_CRISIS = 2      # Last 5 minutes, high surprise
    EPHEMERAL_CONTEXT = 1  # Last 60s, normal operations
    NOISE = 0              # Candidate for pruning

@dataclass
class BridgeMetadata:
    """Tracks which memories serve as cross-domain bridges"""
    memory_id: str
    domain_pairs: Set[Tuple[str, str]]  # e.g., {('physics', 'governance')}
    correlation_strength: float
    last_accessed: datetime
    bridge_score: float = 0.0

    def update_score(self, new_correlation: float):
        """Dynamic bridge scoring"""
        recency = np.exp(-0.1 * (datetime.now() - self.last_accessed).days)
        self.correlation_strength = 0.7 * self.correlation_strength + 0.3 * new_correlation
        self.bridge_score = self.correlation_strength * recency * len(self.domain_pairs)
        self.last_accessed = datetime.now()

class TemporalWeightParams:
    """Temporal decay with ethical and surprise modifiers"""
    base_decay_lambda: float = 0.05  # λ in e^(-λΔt)
    ethical_boost: float = 1.5       # Multiplier for high-β memories
    surprise_boost: float = 2.0      # Multiplier for high-surprise memories
    anchor_permanence: float = 0.3   # Minimum weight for KARNAK-sealed

class TemporalAttentionWithShadow:
    """
    MAT v2: Preserves cross-domain bridges while pruning noise
    Implements the Λ(t) = e^(-λΔt) + ρ·Φ + γ·S + δ·B
    """

    def __init__(self, memory_factory, graph_navigator, latent_linker, initial_lambda: float = 0.05):
        self.factory = memory_factory
        self.navigator = graph_navigator
        self.linker = latent_linker

        # Core MAT
        self.lambda_controller = DynamicLambdaControllerWithDiscovery(baseline=initial_lambda)
        self.params = TemporalWeightParams()
        self.params.base_decay_lambda = initial_lambda

        # Shadow Graph System
        self.shadow_graph = nx.Graph()
        self.bridge_registry: Dict[str, BridgeMetadata] = {}
        self.shadow_retention_days = 14
        self.last_pruning_time = datetime.now()

        # Caches
        self.weight_cache: Dict[str, float] = {}
        self.priority_cache: Dict[str, MemoryPriority] = {}

        # Statistics
        self.stats = {
            'active_memories': 0,
            'shadow_memories': 0,
            'bridges_preserved': 0,
            'false_prunes': 0,
            'bridge_discoveries': 0
        }

    def compute_temporal_weight(self,
                                          memory_id: str,
                                          current_time: datetime) -> float:
        """
        Enhanced Λ(t) = e^(-λΔt) + ρ·Φ + γ·S + δ·B
        """
        manifest = self.factory.anchor_registry.get(memory_id)
        if not manifest: return 0.0

        # Calculate Δt in hours
        delta_t = (current_time - datetime.fromtimestamp(manifest.creation_timestamp)).total_seconds() / 3600.0

        # Base exponential decay
        lambda_dynamic = self.lambda_controller.current_lambda
        recency = np.exp(-lambda_dynamic * delta_t)

        # Permanence component for KARNAK-sealed anchors
        permanence = 0.0
        if manifest.karnak_seal:
            centrality = self._calculate_centrality(memory_id)
            permanence = self.params.anchor_permanence * centrality * manifest.phi_coherence

        # Surprise component
        surprise = 0.0
        if hasattr(self.linker, 'correlation_history'):
            surprises = [c.similarity_score for c in self.linker.correlation_history if memory_id in [c.source_id, c.target_id]]
            surprise = self.params.surprise_boost * np.mean(surprises) if surprises else 0.0

        # Ethical boost
        ethical = self.params.ethical_boost * manifest.benevolence_index if manifest.benevolence_index > 0.75 else 0.0

        # Bridge preservation bonus
        bridge_bonus = 0.0
        if memory_id in self.bridge_registry:
            bridge_bonus = 0.25 * self.bridge_registry[memory_id].bridge_score

        # Active cross-domain bonus
        if self._is_active_cross_domain_link(memory_id):
            bridge_bonus += 0.15

        # Causal depth bonus
        causal_bonus = 0.1 * len(self.navigator.find_causal_path(memory_id, max_depth=10))

        final_weight = recency + permanence + surprise + ethical + bridge_bonus + causal_bonus
        weight = min(final_weight, 1.0)

        self.weight_cache[memory_id] = weight
        self.priority_cache[memory_id] = self._classify_priority(weight, manifest, delta_t)

        return weight

    def _calculate_centrality(self, memory_id: str) -> float:
        if memory_id not in self.navigator.dag: return 0.5
        degree = len(list(self.navigator.dag.predecessors(memory_id))) + len(list(self.navigator.dag.successors(memory_id)))
        return min(degree / 20.0, 1.0)

    def _is_active_cross_domain_link(self, memory_id: str) -> bool:
        if not hasattr(self.linker, 'correlation_history'): return False
        recent_cutoff = time.time() - 86400 # 24h
        for corr in self.linker.correlation_history:
            if corr.discovery_timestamp < recent_cutoff: continue
            if memory_id in [corr.source_id, corr.target_id]:
                # Heuristic: different domains
                dom1 = self._infer_domain(corr.source_id)
                dom2 = self._infer_domain(corr.target_id)
                if dom1 != dom2: return True
        return False

    def _classify_priority(self, weight: float, manifest, delta_t: float) -> MemoryPriority:
        if manifest.karnak_seal and weight > 0.7: return MemoryPriority.ANCHOR_MASTER
        if manifest.benevolence_index > 0.75 and delta_t < 24: return MemoryPriority.ETHICAL_ANCHOR
        if delta_t < 0.083 and manifest.crisis_ratio > 0.5: return MemoryPriority.RECENT_CRISIS
        if delta_t < 1.0: return MemoryPriority.EPHEMERAL_CONTEXT
        return MemoryPriority.NOISE

    async def perform_intelligent_pruning(self, current_time: datetime, target_reduction: float = 0.2):
        if (current_time - self.last_pruning_time).total_seconds() < 3600: return 0

        all_ids = list(self.factory.anchor_registry.keys())
        memory_data = []
        for mid in all_ids:
            weight = self.compute_temporal_weight(mid, current_time)
            manifest = self.factory.anchor_registry[mid]
            memory_data.append({
                'id': mid, 'weight': weight, 'is_bridge': mid in self.bridge_registry,
                'beta': manifest.benevolence_index,
                'bridge_score': self.bridge_registry[mid].bridge_score if mid in self.bridge_registry else 0.0
            })

        memory_data.sort(key=lambda x: self._pruning_priority_score(x))
        target_keep = int(len(all_ids) * (1 - target_reduction))
        pruned = 0
        for m in memory_data[:max(0, len(all_ids) - target_keep)]:
            if m['beta'] > 0.7 or (m['is_bridge'] and m['bridge_score'] > 0.5):
                self.stats['bridges_preserved'] += 1
                continue
            await self._move_to_shadow(m['id'])
            pruned += 1

        self.last_pruning_time = current_time
        return pruned

    def _pruning_priority_score(self, m: Dict) -> float:
        score = 1.0 - m['weight']
        if m['is_bridge']: score -= m['bridge_score'] * 0.5
        if m['beta'] > 0.7: score -= 0.3
        if self._is_active_cross_domain_link(m['id']): score -= 0.2
        return max(score, 0.0)

    async def _move_to_shadow(self, memory_id: str):
        if memory_id not in self.factory.anchor_registry: return
        manifest = self.factory.anchor_registry.pop(memory_id)
        self.shadow_graph.add_node(memory_id, manifest=manifest, moved_at=datetime.now(), original_weight=self.weight_cache.get(memory_id, 0.0))
        if memory_id in self.navigator.dag: self.navigator.dag.remove_node(memory_id)
        self.weight_cache.pop(memory_id, None)
        self.priority_cache.pop(memory_id, None)

    def update_bridge_registry(self, corr):
        for mid in [corr.source_id, corr.target_id]:
            pair = tuple(sorted([self._infer_domain(corr.source_id), self._infer_domain(corr.target_id)]))
            if mid not in self.bridge_registry:
                self.bridge_registry[mid] = BridgeMetadata(mid, {pair}, corr.similarity_score, datetime.now())
            else:
                self.bridge_registry[mid].domain_pairs.add(pair)
                self.bridge_registry[mid].update_score(corr.similarity_score)
        self.stats['bridge_discoveries'] += 1

    def _infer_domain(self, mid: str) -> str:
        manifest = self.factory.anchor_registry.get(mid) or (self.shadow_graph.nodes[mid]['manifest'] if mid in self.shadow_graph else None)
        return manifest.substrate_domain if manifest else 'unknown'

    def query_cross_domain_bridges(self, domain_a: str, domain_b: str, min_correlation: float = 0.6) -> List[Dict]:
        bridges = []
        target_pair = tuple(sorted([domain_a, domain_b]))
        for mid, bridge in self.bridge_registry.items():
            if target_pair in bridge.domain_pairs and bridge.correlation_strength >= min_correlation:
                tier = 'ACTIVE' if mid in self.factory.anchor_registry else 'SHADOW'
                bridges.append({'memory_id': mid, 'tier': tier, 'bridge_score': bridge.bridge_score, 'correlation': bridge.correlation_strength})
        return sorted(bridges, key=lambda x: x['bridge_score'], reverse=True)

class DynamicLambdaControllerWithDiscovery:
    def __init__(self, min_lambda=0.01, max_lambda=0.2, baseline=0.05):
        self.min_lambda, self.max_lambda, self.current_lambda = min_lambda, max_lambda, baseline
    def adjust(self, system_load, ethical_stability, surprise_rate, linker_mode='exploration'):
        adjustment = 0.5 * system_load + 0.3 * (1.0 - ethical_stability) + 0.2 * surprise_rate
        new_lambda = self.current_lambda * (0.7 + 0.3 * adjustment)
        if linker_mode == 'exploration': new_lambda *= 0.6
        self.current_lambda = max(self.min_lambda, min(self.max_lambda, 0.8 * self.current_lambda + 0.2 * new_lambda))
        return self.current_lambda
