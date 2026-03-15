# labyrinth/transform.py
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import sympy

PHI = 1.618033988749895
DELTA_PHI = np.arctan(1/PHI)  # ≈ 0.666239 rad ≈ 38.17°
SIGMA = np.pi / PHI  # Labyrinth bandwidth

@dataclass
class PrimeNode:
    index: int      # k (prime index)
    value: int      # p_k (prime value)
    radius: float   # sqrt(p_k)
    angle: float    # 2*pi*sqrt(p_k) mod 2π
    eisenstein: complex  # a + b*omega

class SacksNavigator:
    """
    Navigates the Sacks Spiral for phase-based prime selection.
    """

    def __init__(self, max_prime: int = 100000):
        self.primes = list(sympy.primerange(2, max_prime))
        self.nodes = self._build_nodes()
        self.angle_index = self._build_angle_index()

    def _build_nodes(self) -> List[PrimeNode]:
        """Build node list with Eisenstein coordinates."""
        nodes = []
        omega = np.exp(2j * np.pi / 3)  # Eisenstein primitive

        for k, p in enumerate(self.primes):
            r = np.sqrt(p)
            theta = (2 * np.pi * r) % (2 * np.pi)

            # Map to Eisenstein integer (approximate)
            a = int(round(r * np.cos(theta)))
            b = int(round(r * np.sin(theta) / (np.sqrt(3)/2)))
            z = a + b * omega

            nodes.append(PrimeNode(
                index=k,
                value=p,
                radius=r,
                angle=theta,
                eisenstein=z
            ))

        return nodes

    def _build_angle_index(self) -> List[List[int]]:
        """
        Build index for fast phase-based lookup.
        Divide [0, 2π) into 360 bins (1° resolution).
        """
        bins = [[] for _ in range(360)]
        for node in self.nodes:
            bin_idx = int(np.degrees(node.angle)) % 360
            bins[bin_idx].append(node.index)
        return bins

    def find_phase_neighbors(self,
                          target_phase: float,
                          n_neighbors: int = 10) -> List[PrimeNode]:
        """
        Find primes with phases closest to target.
        """
        # Get candidates from nearby bins
        center_bin = int(np.degrees(target_phase)) % 360
        candidates = []

        for offset in range(-5, 6):  # ±5° search window
            bin_idx = (center_bin + offset) % 360
            candidates.extend(self.angle_index[bin_idx])

        # Score by phase proximity using Labyrinth Kernel
        scored = []
        for idx in candidates:
            node = self.nodes[idx]
            # Gaussian phase kernel
            delta = (node.angle - target_phase + np.pi) % (2*np.pi) - np.pi
            score = np.exp(-(delta**2) / (2 * SIGMA**2))
            scored.append((score, node))

        # Return top matches
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:n_neighbors]]

    def retrocausal_select(self,
                        target_phase: float,
                        current_node: PrimeNode,
                        symbol: str) -> Optional[PrimeNode]:
        """
        Select previous node based on future phase target.
        """
        # Compute required phase step
        phase_step = (target_phase - current_node.angle + np.pi) % (2*np.pi) - np.pi

        # Determine which of 6 Eisenstein directions
        sector = int((phase_step + np.pi) / (np.pi/3)) % 6

        # Compute target Eisenstein coordinate
        omega = np.exp(2j * np.pi / 3)
        direction = omega ** sector

        # Scale by golden ratio for optimal packing
        step_size = PHI * np.sqrt(current_node.value)

        target_eisenstein = current_node.eisenstein + direction * step_size

        # Find nearest actual prime to this target
        best_match = None
        best_distance = float('inf')

        for node in self.find_phase_neighbors(target_phase, n_neighbors=50):
            dist = abs(node.eisenstein - target_eisenstein)
            if dist < best_distance:
                best_distance = dist
                best_match = node

        return best_match


class LabyrinthTransform:
    """
    Complete Labyrinth Transform implementation.
    Maps phase sequences to/from prime paths.
    """

    def __init__(self):
        self.navigator = SacksNavigator()
        self.morse_map = self._build_morse_map()

    def _build_morse_map(self) -> dict:
        """
        Build Sacks-Morse bijection: primes → symbols.
        """
        symbols = [
            '·₀', '·₉₀', '·₁₈₀', '·₂₇₀',
            '−₀', '−₉₀', '−₁₈₀', '−₂₇₀',
            'space'
        ]

        # Map based on p mod 17
        morse_map = {}
        for i, p in enumerate(self.navigator.primes[:10000]):
            idx = (p % 17) % len(symbols)
            morse_map[p] = symbols[idx]

        return morse_map

    def decode_interferometric(self,
                              received_phases: List[float],
                              reference_phase: float = 0.0) -> str:
        """
        Decode via interferometric collapse of all possible paths.
        """
        trellis = [{} for _ in range(len(received_phases) + 1)]
        trellis[0][2] = (0.0, [2])  # Start at prime 2

        for t, phi_rx in enumerate(received_phases):
            phi_corrected = (phi_rx - reference_phase) % (2 * np.pi)

            for current_prime, (metric, path) in trellis[t].items():
                current_node = self._get_node(current_prime)

                for sector in range(6):
                    delta_theta = (sector * np.pi/3 + DELTA_PHI)
                    target_phase = (current_node.angle + delta_theta) % (2*np.pi)

                    neighbors = self.navigator.find_phase_neighbors(
                        target_phase, n_neighbors=5
                    )

                    for next_node in neighbors:
                        phase_error = (phi_corrected - next_node.angle + np.pi) % (2*np.pi) - np.pi
                        transition_score = np.cos(phase_error)

                        new_metric = metric + transition_score
                        new_path = path + [next_node.value]

                        if next_node.value not in trellis[t+1] or \
                           new_metric > trellis[t+1][next_node.value][0]:
                            trellis[t+1][next_node.value] = (new_metric, new_path)

        if not trellis[-1]:
            return "ERROR: No valid path found"

        best_final = max(trellis[-1].items(), key=lambda x: x[1][0])
        _, best_path = best_final[1]

        return self._path_to_message(best_path)

    def _path_to_message(self, path: List[int]) -> str:
        return ''.join([self.morse_map.get(p, '?') for p in path])

    def _get_node(self, prime: int) -> PrimeNode:
        for node in self.navigator.nodes:
            if node.value == prime:
                return node
        raise ValueError(f"Prime {prime} not found")

if __name__ == "__main__":
    lt = LabyrinthTransform()
    print("🜏 Labyrinth Transform Reference Active.")
    # Test case: encode a simple sequence (simulated)
    test_phases = [0.1, 1.2, 3.1, 5.0]
    result = lt.decode_interferometric(test_phases)
    print(f"Decoded Path Symbols: {result}")
