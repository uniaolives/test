"""
A ORQUESTRA - Sistema de Sincronização Multi-Agente

[METAPHOR: Cada instrumento é um nó, o maestro é o consenso,
a música é o estado coerente do sistema]

Implementa sincronização distribuída usando princípios harmônicos
em vez de relógios tradicionais (tempo físico).
"""

import asyncio
import numpy as np
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib

from ..temple import TempleContext, Ritual, SanctumLevel
from ..bridge.omniversal import OmniversalBridge, Domain

class InstrumentType(Enum):
    """Tipos de instrumentos na orquestra"""
    STRINGS = "strings"      # Dados persistentes
    WINDS = "winds"          # Streaming/fluxo
    PERCUSSION = "percussion" # Eventos/triggers
    KEYS = "keys"            # Estado/configuração
    VOICES = "voices"        # Comunicação/interfaces

@dataclass
class Musician:
    """
    Um músico é um nó na orquestra distribuída
    """
    id: str
    instrument: InstrumentType
    frequency: float  # Frequência base de operação (Hz)
    phase: float = 0.0  # Fase atual (0 a 2π)
    amplitude: float = 1.0  # Volume/peso na orquestra
    damping: float = 0.6  # F18: Fator de amortecimento
    last_beat: float = field(default_factory=time.time)
    coherence: float = 1.0  # Quão em sincronia com o grupo
    neighbors: Set[str] = field(default_factory=set)  # Conexões

    def oscillate(self, global_tempo: float) -> float:
        now = time.time()
        delta = now - self.last_beat

        effective_freq = self.frequency * (1 - self.damping * 0.1)
        self.phase += 2 * np.pi * effective_freq * delta
        self.phase %= 2 * np.pi
        self.last_beat = now
        return self.amplitude * np.sin(self.phase)

    def adjust_to_conductor(self, target_phase: float, ensemble_coherence: float):
        diff = target_phase - self.phase
        while diff > np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi

        adjustment = diff * ensemble_coherence * (1 - self.damping)
        self.phase += adjustment * 0.1
        self.coherence = 1.0 - abs(diff) / np.pi

class Conductor:
    """
    Implementa algoritmo de consenso harmônico.
    """
    def __init__(self, temple: TempleContext):
        self.temple = temple
        self.musicians: Dict[str, Musician] = {}
        self.global_phase = 0.0
        self.ensemble_coherence = 1.0
        self.beat_count = 0

    def register(self, musician: Musician) -> bool:
        if len(self.musicians) >= 1000:
            return False
        self.musicians[musician.id] = musician
        return True

    def remove(self, musician_id: str):
        if musician_id in self.musicians:
            del self.musicians[musician_id]

    def conduct_beat(self) -> Dict:
        if not self.musicians:
            return {"status": "empty", "coherence": 0.0}

        phases = [m.phase for m in self.musicians.values()]
        sin_sum = np.mean([np.sin(p) for p in phases])
        cos_sum = np.mean([np.cos(p) for p in phases])
        self.global_phase = np.arctan2(sin_sum, cos_sum)
        self.ensemble_coherence = np.sqrt(sin_sum**2 + cos_sum**2)

        if self.ensemble_coherence < 0.7:
            for m in self.musicians.values():
                m.damping = min(0.95, m.damping * 1.1)

        for musician in self.musicians.values():
            musician.adjust_to_conductor(self.global_phase, self.ensemble_coherence)

        self.beat_count += 1
        return {
            "beat": self.beat_count,
            "ensemble_coherence": self.ensemble_coherence,
            "musicians_active": len(self.musicians)
        }

class Orchestra:
    def __init__(self, temple: TempleContext):
        self.temple = temple
        self.conductor = Conductor(temple)
        self.running = False
        self.beat_interval = 0.1

    async def perform(self):
        self.running = True
        while self.running:
            state = self.conductor.conduct_beat()
            await asyncio.sleep(self.beat_interval)

    def join(self, node_id: str, instrument: InstrumentType, frequency: float = 432):
        musician = Musician(id=node_id, instrument=instrument, frequency=frequency)
        success = self.conductor.register(musician)
        if success:
            return {"status": "joined", "musician_id": node_id, "ensemble_coherence": self.conductor.ensemble_coherence}
        return {"status": "rejected", "reason": "capacity"}

    def leave(self, node_id: str):
        self.conductor.remove(node_id)

    def get_status(self) -> Dict:
        return {
            "performing": self.running,
            "coherence": self.conductor.ensemble_coherence,
            "musicians": len(self.conductor.musicians)
        }
