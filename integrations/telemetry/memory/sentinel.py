# sentinel.py
# Real-time Monitoring for CS2↔AoE Bridge Preservation
# Memory ID 37-A1: Critical Bridge Sentinel

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass
class CriticalBridge:
    """A bridge deemed critical for cross-domain intuition"""
    memory_id: str
    physics_signature: str
    governance_impact: str
    discovery_timestamp: datetime
    correlation_strength: float
    pruning_risk: float = 0.0

class BridgeSentinel:
    """
    Real-time monitor that tracks CS2↔AoE bridges during validation.
    """

    CRITICAL_SIGNATURES = {
        "high_velocity_variance": {"expected_impact": "attention_instability", "risk_threshold": 0.7, "min_preservation_hours": 72},
        "collision_cascade": {"expected_impact": "cascade_failure_economic", "risk_threshold": 0.8, "min_preservation_hours": 96},
        "network_lag_spike": {"expected_impact": "productivity_drop", "risk_threshold": 0.6, "min_preservation_hours": 48},
        "fp16_precision_loss": {"expected_impact": "resource_allocation_error", "risk_threshold": 0.75, "min_preservation_hours": 60}
    }

    def __init__(self, mat_shadow_system):
        self.mat = mat_shadow_system
        self.critical_bridges: Dict[str, CriticalBridge] = {}
        self.risk_log = []
        self.preservation_log = []
        self.stats = {'critical_bridges_found': 0, 'at_risk_bridges': 0, 'premature_prunes_prevented': 0, 'false_positive_alerts': 0}

    async def scan_critical_bridges(self) -> List[CriticalBridge]:
        bridges = self.mat.query_cross_domain_bridges('PHYSICS', 'GOVERNANCE')
        new_critical = []
        for b in bridges:
            mid = b['memory_id']
            if mid in self.critical_bridges: continue

            manifest = self.mat.factory.anchor_registry.get(mid) or (self.mat.shadow_graph.nodes[mid]['manifest'] if mid in self.mat.shadow_graph else None)
            if not manifest: continue

            sig = self._identify_physics_signature(manifest)
            impact = self._identify_governance_impact(manifest)

            if sig in self.CRITICAL_SIGNATURES:
                cb = CriticalBridge(mid, sig, impact, datetime.now(), b['correlation'])
                self.critical_bridges[mid] = cb
                new_critical.append(cb)
        self.stats['critical_bridges_found'] += len(new_critical)
        return new_critical

    def _identify_physics_signature(self, manifest) -> str:
        # Simplified detection for the sentinel
        domain = getattr(manifest, 'substrate_domain', '')
        if domain != 'PHYSICS': return 'unknown'
        latency = getattr(manifest, 'latency_p99', 0.0)
        if latency > 9.0: return 'network_lag_spike'
        return 'high_velocity_variance'

    def _identify_governance_impact(self, manifest) -> str:
        crisis = getattr(manifest, 'crisis_ratio', 0.0)
        if crisis > 0.8: return 'cascade_failure_economic'
        return 'attention_instability'

    async def assess_pruning_risk(self) -> List[Tuple[str, float]]:
        at_risk = []
        now = datetime.now()
        for mid, bridge in self.critical_bridges.items():
            weight = self.mat.compute_temporal_weight(mid, now)
            tier = 'ACTIVE' if mid in self.mat.factory.anchor_registry else 'SHADOW'
            hours = (now - bridge.discovery_timestamp).total_seconds() / 3600

            risk = (0.4 * (max(0, 0.4 - weight) / 0.4) +
                    0.3 * (0.3 if tier == 'SHADOW' else 0.0) +
                    0.2 * (1.0 - min(hours / 48.0, 1.0)))

            bridge.pruning_risk = risk
            if risk > 0.5: at_risk.append((mid, risk))
        self.stats['at_risk_bridges'] = len(at_risk)
        return at_risk

    async def apply_protective_interventions(self, at_risk: List[Tuple[str, float]]):
        for mid, risk in at_risk:
            if risk > 0.7:
                if mid in self.mat.bridge_registry:
                    self.mat.bridge_registry[mid].bridge_score = min(1.0, self.mat.bridge_registry[mid].bridge_score + 0.3)
                    self.stats['premature_prunes_prevented'] += 1

    def generate_report(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.now().isoformat(),
            'total_critical': len(self.critical_bridges),
            'stats': self.stats
        }
