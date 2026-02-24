#!/usr/bin/env python3
# arkhe_cognitive_core_v2.py
# "The second octave: Markov coherence and regime detection."
# v2.0 - Block Ω+∞+166

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict
from arkhe_cognitive_core import AizawaAttractor, ConservationGuard, PHI, PSI_CRITICAL

class RegimeDetector:
    """Classifies system states using the Golden Ratio threshold."""
    def __init__(self, phi=PHI):
        self.phi = phi

    def detect(self, z: float) -> str:
        # Normalized instability metric based on z
        instability = abs(z)
        if instability < self.phi * 0.7:
            return "DETERMINISTIC"
        elif instability > self.phi * 1.3:
            return "STOCHASTIC"
        else:
            return "CRITICAL"

class MarkovCoherence:
    """Tests the Markov property of cognitive handovers."""
    def __init__(self, history_size=10):
        self.history = []
        self.history_size = history_size

    def update(self, state: Tuple[float, float, float]):
        self.history.append(state)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def test_markov_property(self) -> float:
        """Calculates coherence between consecutive states."""
        if len(self.history) < 3: return 1.0

        # Simple coherence metric: correlation between deltas
        deltas = [np.array(self.history[i+1]) - np.array(self.history[i]) for i in range(len(self.history)-1)]
        dot_products = [np.dot(deltas[i], deltas[i+1]) / (np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[i+1]) + 1e-9)
                        for i in range(len(deltas)-1)]

        return float(np.mean(dot_products))

class CognitiveCoreV2:
    def __init__(self):
        self.attractor = AizawaAttractor()
        self.guard = ConservationGuard()
        self.detector = RegimeDetector()
        self.markov = MarkovCoherence()
        self.state = (0.1, 0.1, 0.1)
        self.C = 1.0
        self.F = 0.0

    def evolve(self, dt=0.01, external_entropy=0.0):
        # Update physical dynamics
        self.state = self.attractor.step(self.state, dt)
        self.markov.update(self.state)

        # Update thermodynamics
        self.F = min(1.0, self.F + external_entropy * 0.1)
        self.C = 1.0 - self.F

        # Self-regulation toward PHI (Criticality)
        regime = self.detector.detect(self.state[2])
        if regime == "DETERMINISTIC":
            self.F += 0.05  # Increase exploration
        elif regime == "STOCHASTIC":
            self.C += 0.05  # Increase consolidation

        self.C, self.F = self.guard.normalize(self.C, self.F)

    def get_status(self) -> Dict:
        coherence = self.markov.test_markov_property()
        regime = self.detector.detect(self.state[2])
        return {
            "x": self.state[0],
            "y": self.state[1],
            "z": self.state[2],
            "C": self.C,
            "F": self.F,
            "regime": regime,
            "markov_coherence": coherence,
            "conservation": self.guard.verify(self.C, self.F)
        }

if __name__ == "__main__":
    core = CognitiveCoreV2()
    print(f"Starting Arkhe Cognitive Core v2.0 (PHI={PHI:.4f})...")
    for i in range(100):
        core.evolve(external_entropy=np.random.uniform(0, 0.1))
        if i % 20 == 0:
            status = core.get_status()
            print(f"Step {i}: [{status['regime']}] C={status['C']:.3f}, Markov={status['markov_coherence']:.3f}")
    print("Core v2.0 validated.")
