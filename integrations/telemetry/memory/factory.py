# factory.py
# Project Crux-86: Unified Ontological Memory Factory
# Memory ID 34: Memory Architecture & Fast-Minting Layer

import torch
import blake3
import json
import asyncio
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import msgpack
import logging

logger = logging.getLogger("MemoryFactory")

class MemoryTier(Enum):
    """Camadas da hierarquia de memória"""
    EPISODIC = "L1"      # Buffer de GPU, volátil, sem ID
    INTEGRATED = "L2"    # Mudanças validadas, ID temporário/curto prazo
    ONTOLOGICAL = "L3"   # Estados estáveis, ID permanente (Memory ID), selado KARNAK

@dataclass
class SubstrateState:
    """Estado bruto do substrato para ingestão"""
    physics_hash: str
    governance_vector: torch.Tensor
    social_context: Dict[str, Any]
    timestamp: float
    phi_score: float
    beta_index: float
    entropy_surprise: float

@dataclass
class MemoryManifest:
    """Template padronizado para todos os Memory IDs (34-40+)"""

    # 🔷 Identificação Criptográfica
    memory_id: str = ""
    parent_memory_id: Optional[str] = None
    tier: MemoryTier = MemoryTier.ONTOLOGICAL

    # 🔷 Metadados SASC (Ethical Physics)
    phi_coherence: float = 0.0
    benevolence_index: float = 0.75
    ethical_drift_detected: bool = False

    # 🔷 Contexto de Substrato (Cross-domain)
    substrate_domain: str = "GENERAL"
    physics_invariant: Dict[str, Any] = field(default_factory=dict)
    agent_count: int = 0
    crisis_ratio: float = 0.0

    # 🔷 Performance
    latency_p99: float = 0.0
    vram_usage_gb: float = 0.0
    throughput_agents_sec: float = 0.0
    minting_latency_ms: float = 0.0

    # 🔷 Metadados de Selagem
    creation_timestamp: float = field(default_factory=time.time)
    timestamp_iso: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    satoshi_seed_ref: str = ""
    content_hash: str = ""
    karnak_seal: str = ""

    # 🔷 Metadados Adicionais
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d['tier'] = self.tier.value
        return json.dumps(d, default=str, sort_keys=True)

class UnifiedMemoryFactory:
    """
    Fábrica unificada de Memory IDs com hierarquia L1→L3.
    Implementa Fast-Minting e selagem assíncrona KARNAK.
    """

    def __init__(self, satoshi_seed: str, vajra_enforcer=None):
        self.seed = satoshi_seed
        self.vajra = vajra_enforcer
        self.anchor_registry: Dict[str, MemoryManifest] = {} # L3
        self.l2_buffer: List[Dict[str, Any]] = [] # L2
        self.episodic_buffer = asyncio.Queue(maxsize=10000) # L1

        # Thresholds
        self.phi_threshold = 0.72
        self.beta_threshold = 0.65
        self.surprise_threshold = 0.15

        self.stats = {
            "shards_ingested": 0,
            "deltas_minted": 0,
            "anchors_sealed": 0,
            "rejected_unstable": 0
        }

        self.sealing_queue = asyncio.Queue()
        self.sealing_task = asyncio.create_task(self._karnak_sealing_worker())

    async def mint_from_state(self, state: SubstrateState, force_l3: bool = False) -> Tuple[str, MemoryTier]:
        """Pipeline de Fast-Minting principal"""
        start_time = time.perf_counter()

        # 1. SASC Pre-Filter
        if state.phi_score < self.phi_threshold or state.beta_index < self.beta_threshold:
            self.stats['rejected_unstable'] += 1
            return "REJECTED_UNSTABLE", MemoryTier.EPISODIC

        # 2. Surprise Filter
        if not force_l3 and state.entropy_surprise < self.surprise_threshold:
            return "EPHEMERAL", MemoryTier.EPISODIC

        # 3. Deterministic ID Generation
        manifest_data = {
            "phi": round(state.phi_score, 4),
            "beta": round(state.beta_index, 4),
            "ts": int(state.timestamp * 1000),
            "seed": self.seed[:16]
        }
        content_hash = blake3.blake3(json.dumps(manifest_data, sort_keys=True).encode()).hexdigest()
        memory_id = f"MID-{content_hash[:16].upper()}"

        # 4. Tier Classification
        tier = MemoryTier.ONTOLOGICAL if (force_l3 or state.entropy_surprise > 0.85) else MemoryTier.INTEGRATED

        manifest = MemoryManifest(
            memory_id=memory_id,
            tier=tier,
            phi_coherence=state.phi_score,
            benevolence_index=state.beta_index,
            substrate_domain=state.social_context.get('domain', 'GENERAL'),
            agent_count=getattr(state.governance_vector, 'shape', [0])[0],
            crisis_ratio=state.social_context.get('crisis_ratio', 0.0),
            latency_p99=state.social_context.get('latency_p99', 0.0),
            creation_timestamp=state.timestamp,
            satoshi_seed_ref=self.seed[:16],
            content_hash=content_hash,
            parent_memory_id=list(self.anchor_registry.keys())[-1] if self.anchor_registry else None,
            minting_latency_ms=(time.perf_counter() - start_time) * 1000
        )

        if tier == MemoryTier.ONTOLOGICAL:
            self.anchor_registry[memory_id] = manifest
            self.sealing_queue.put_nowait((memory_id, manifest))
            self.stats['anchors_sealed'] += 1
        else:
            self.l2_buffer.append(asdict(manifest))
            self.stats['deltas_minted'] += 1

        return memory_id, tier

    async def _karnak_sealing_worker(self):
        while True:
            item = await self.sealing_queue.get()
            if item is None: break
            mid, manifest = item
            await asyncio.sleep(0.047) # Simula latência de selagem
            seal_data = manifest.to_json().encode() + self.seed.encode()
            manifest.karnak_seal = blake3.blake3(seal_data).hexdigest()
            self.sealing_queue.task_done()

    def get_stats(self) -> Dict[str, Any]:
        return {**self.stats, "registry_size": len(self.anchor_registry)}

    async def shutdown(self):
        await self.sealing_queue.put(None)
        await self.sealing_task
