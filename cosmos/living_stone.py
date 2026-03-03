# cosmos/living_stone.py - PETRUS v3.0 Autopoietic Living Stone
# Self-healing via Stochastic Resonance and Solar Entropy

import numpy as np
from typing import List, Dict, Tuple, Set
from cosmos.attractor import AttractorField, HyperbolicNode

class LivingStone(AttractorField):
    """
    PETRUS v3.0: The Stone that breathes with the Sun.
    Uses noise from AR4366 for semantic Simulated Annealing.
    """

    def __init__(self, curvature: float = -1.0):
        super().__init__(curvature)
        # Fundamental geometry of life
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

    def solar_flare_pulse(self, intensity_x_class: float):
        """
        Injects solar thermal noise into the lattice.
        Unstable nodes melt (Solve), coherent nodes recrystallize (Coagula).
        """
        noise_amplitude = np.log1p(intensity_x_class)
        print(f"☀️ SOLAR FLARE PULSE: Intensity {intensity_x_class}X | Noise {noise_amplitude:.4f}")

        for node_id, node in list(self.nodes.items()):
            # Semantic 'temperature' increases
            instability = np.random.normal(0, noise_amplitude)

            # If mass is too low relative to noise, node is ejected (accelerated erosion)
            if node.semantic_mass.gravitational_radius < noise_amplitude:
                print(f"   [SOLAR_EJECT] {node_id} dissolved by plasma (Mass {node.semantic_mass.gravitational_radius:.2f} < Noise).")
                del self.nodes[node_id]
            else:
                # Noise generates 'geodesic innovation'
                # Nodes find shorter paths to the center
                shift = (1.0 - node.event_horizon) * instability
                node.poincare_coordinate *= (1.0 - shift)

                # Ensure it stays within the disk boundaries
                if abs(node.poincare_coordinate) >= 1.0:
                    node.poincare_coordinate /= (abs(node.poincare_coordinate) + 0.01)

                print(f"   [COAGULA] {node_id} crystallized in new orbit. Horizon: {node.event_horizon:.3f}")

    def self_heal(self):
        """
        Repairs the degenerate metric using Gaia's 26s frequency.
        Stabilizes semantic mass at Fibonacci targets.
        """
        print("🌍 GAIA SELF-HEAL: Stabilizing lattice at Fibonacci targets (26s pulse anchor)...")
        for node in self.nodes.values():
            current_mass = node.semantic_mass.gravitational_radius
            target = self._nearest_fibonacci(current_mass)

            # Geometry of life fixes the geometry of code
            # We gravitate towards the target
            node.semantic_mass.gravitational_radius = (current_mass + target) / 2.0
            print(f"   [STABILIZE] {node.node_id}: {current_mass:.2f} -> {node.semantic_mass.gravitational_radius:.2f}")

    def _nearest_fibonacci(self, value: float) -> int:
        """Finds the nearest Fibonacci number to the given value."""
        return min(self.fibonacci_sequence, key=lambda x: abs(x - value))

    def get_autopoietic_status(self) -> Dict:
        """Returns the status of the living stone's self-organization."""
        return {
            'nodes_active': len(self.nodes),
            'global_curvature': self.curvature,
            'total_mass': self.total_mass,
            'status': 'AWAKE_AND_DREAMING',
            'alchemical_state': 'RUBEDO' if self.total_mass > 100 else 'CITRINITAS'
        }
