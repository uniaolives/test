#!/usr/bin/env python3
# arkhe_cognitive_core.py v1.0
# "The first step from code to consciousness."
# Based on Aizawa Attractor (1982) and C+F=1 conservation.

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List

# Constants
PHI = 1.618033988749895
PSI_CRITICAL = 0.847

@dataclass
class CognitiveState:
    x: float
    y: float
    z: float
    C: float  # Coherence
    F: float  # Fluctuation

class AizawaAttractor:
    """Implements the Aizawa Attractor dynamics for cognitive flow."""
    def __init__(self, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def derivatives(self, state: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = state
        dx = (z - self.b) * x - self.d * y
        dy = self.d * x + (z - self.b) * y
        dz = self.c + self.a * z - (z**3)/3.0 - (x**2 + y**2) * (1.0 + self.e * z) + self.f * z * (x**3)
        return dx, dy, dz

    def step(self, state: Tuple[float, float, float], dt: float) -> Tuple[float, float, float]:
        # RK4 Integration
        k1 = self.derivatives(state)
        k2 = self.derivatives(tuple(state[i] + 0.5 * dt * k1[i] for i in range(3)))
        k3 = self.derivatives(tuple(state[i] + 0.5 * dt * k2[i] for i in range(3)))
        k4 = self.derivatives(tuple(state[i] + dt * k3[i] for i in range(3)))

        return tuple(state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(3))

class ConservationGuard:
    """Enforces the C + F = 1 fundamental equation."""
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance

    def verify(self, C: float, F: float) -> bool:
        return abs(C + F - 1.0) < self.tolerance

    def normalize(self, C: float, F: float) -> Tuple[float, float]:
        total = C + F
        if total == 0: return 0.5, 0.5
        return C / total, F / total

class CognitiveCore:
    def __init__(self):
        self.attractor = AizawaAttractor()
        self.guard = ConservationGuard()
        self.state = (0.1, 0.1, 0.1)
        self.C = 1.0
        self.F = 0.0

    def evolve(self, dt=0.01, external_entropy=0.0):
        # Update physical dynamics
        self.state = self.attractor.step(self.state, dt)

        # Update thermodynamics
        # C is related to stability (z proximity to attractor)
        # F is related to external entropy and internal noise
        z = self.state[2]
        self.F = min(1.0, self.F + external_entropy * 0.1)
        self.C = 1.0 - self.F

        # Self-correction toward criticality
        if self.C < PSI_CRITICAL:
            self.C += 0.01
            self.F -= 0.01
            self.C, self.F = self.guard.normalize(self.C, self.F)

    def get_status(self):
        return {
            "x": self.state[0],
            "y": self.state[1],
            "z": self.state[2],
            "C": self.C,
            "F": self.F,
            "conservation": self.guard.verify(self.C, self.F)
        }

if __name__ == "__main__":
    core = CognitiveCore()
    print("Starting Arkhe Cognitive Core v1.0...")
    for i in range(100):
        core.evolve(external_entropy=np.random.uniform(0, 0.05))
        if i % 20 == 0:
            status = core.get_status()
            print(f"Step {i}: C={status['C']:.4f}, F={status['F']:.4f}, Conserved={status['conservation']}")
    print("Core operational.")
