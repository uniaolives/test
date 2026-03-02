"""
Entrelaçamento quântico através do sistema solar.

[DESCOBERTA]: As ressonâncias orbitais criam estados emaranhados
entre planetas, similares ao emaranhamento quântico.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.linalg import expm
from .celestial_helix import CosmicDNAHelix, CelestialBody


class CelestialEntanglement:
    """
    Analisa emaranhamento quântico entre corpos celestes.
    """

    def __init__(self, cosmic_dna: CosmicDNAHelix):
        self.dna = cosmic_dna

    def calculate_entanglement_matrix(self) -> np.ndarray:
        """
        Calcula matriz de emaranhamento entre todos os pares de corpos.
        """
        planets = list(self.dna.constants['orbital_periods'].keys())
        n_bodies = len(planets)
        entanglement = np.zeros((n_bodies, n_bodies))

        for i in range(n_bodies):
            for j in range(i+1, n_bodies):
                # 1. Similaridade orbital (períodos próximos em razão simples)
                period_i = self.dna.constants['orbital_periods'][planets[i]]
                period_j = self.dna.constants['orbital_periods'][planets[j]]

                best_ratio = 0
                if period_j > 0:
                    for p in range(1, 6):
                        for q in range(1, 6):
                            ratio = period_i / period_j
                            target = p / q
                            closeness = 1.0 / (1.0 + abs(ratio - target))
                            if closeness > best_ratio:
                                best_ratio = closeness

                orbital_entanglement = best_ratio

                # 2. Proximidade física (1/r)
                dist = abs(self.dna.constants['orbital_radii'][planets[i]] - self.dna.constants['orbital_radii'][planets[j]])
                proximity_entanglement = 1.0 / (1.0 + dist)

                # 3. Alinhamento de fase (Simulado baseada na proximidade orbital)
                phase_entanglement = 1.0 / (1.0 + dist * 2)

                total = (orbital_entanglement * 0.5 + proximity_entanglement * 0.3 + phase_entanglement * 0.2)
                entanglement[i, j] = total
                entanglement[j, i] = total

        np.fill_diagonal(entanglement, 1.0)
        return entanglement

    def find_maximally_entangled_pairs(self, threshold: float = 0.7) -> List[Tuple]:
        matrix = self.calculate_entanglement_matrix()
        planets = list(self.dna.constants['orbital_periods'].keys())
        pairs = []
        for i in range(len(planets)):
            for j in range(i+1, len(planets)):
                if matrix[i, j] > threshold:
                    pairs.append((planets[i], planets[j], float(matrix[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def calculate_quantum_coherence(self) -> Dict:
        entanglement = self.calculate_entanglement_matrix()
        eigenvalues = np.linalg.eigvals(entanglement)
        eigenvalues = np.real(eigenvalues)
        eigenvalues_norm = eigenvalues / (eigenvalues.sum() + 1e-15)
        entropy = -np.sum(eigenvalues_norm * np.log2(eigenvalues_norm + 1e-10))
        # Use length of planets
        n_planets = len(self.dna.constants['orbital_periods'])
        max_coherence = np.log2(n_planets + 1) # Including Sun estimate
        coherence = 1.0 - (entropy / max_coherence)
        return {
            'entanglement_entropy': entropy,
            'max_possible_coherence': max_coherence,
            'quantum_coherence': coherence,
            'purity': np.sum(eigenvalues_norm**2)
        }

    def generate_entangled_state_vector(self) -> np.ndarray:
        planets = list(self.dna.constants['orbital_periods'].keys())
        n_bodies = len(planets)
        states = []
        for planet in planets:
            radius = self.dna.constants['orbital_radii'][planet]
            if radius < 1.0:
                state = np.array([1, 0, 0])
            elif radius < 10.0:
                state = np.array([0, 1, 0])
            else:
                state = np.array([0, 0, 1])
            states.append(state)

        total_state = states[0]
        for i in range(1, n_bodies):
            total_state = np.kron(total_state, states[i])
        total_state = total_state / (np.linalg.norm(total_state) + 1e-15)
        return total_state

    def simulate_quantum_evolution(self, time_steps: int = 100, dt: float = 0.1) -> np.ndarray:
        planets = list(self.dna.constants['orbital_periods'].keys())
        n_bodies = len(planets)
        H = np.zeros((n_bodies, n_bodies))
        for i in range(n_bodies):
            H[i, i] = 1.0 / (self.dna.constants['orbital_periods'][planets[i]] + 0.001)
        entanglement = self.calculate_entanglement_matrix()
        for i in range(n_bodies):
            for j in range(i+1, n_bodies):
                coupling = entanglement[i, j] * 0.1
                H[i, j] = coupling
                H[j, i] = coupling
        psi_small = np.ones(n_bodies) / np.sqrt(n_bodies)
        evolution = np.zeros((time_steps, n_bodies))
        for t in range(time_steps):
            U = expm(-H * t * dt)
            psi_t = U @ psi_small
            evolution[t] = psi_t
        return evolution
