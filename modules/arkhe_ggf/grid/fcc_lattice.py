# arkhe_ggf/grid/fcc_lattice.py
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from core.python.arkhe_physics.entropy_unit import ArkheEntropyUnit

@dataclass
class STARNode:
    """Representa um nó STAR (Space-Time Array Resonator) no lattice FCC."""
    position: np.ndarray  # Coordenadas (x, y, z) em metros
    angular_momentum: np.ndarray  # Vetor de spin
    scalar_a: float  # Quaternion scalar 'a' (densidade de estresse EM)

    def reemit_photon(self, incoming_wavelength: float) -> float:
        """
        Implementa o Extinction Shift Principle (Dowdye):
        Absorve fóton incidente e reemite com velocidade c relativa ao campo local.
        Retorna novo comprimento de onda (deslocado por Doppler devido à velocidade do nó).
        """
        # Cálculo do deslocamento Doppler devido à velocidade do nó
        v_node = self.angular_momentum / self.scalar_a  # Simplificação
        doppler_shift = (1 - v_node[0] / 299792458.0)  # Apenas componente x para exemplo
        return incoming_wavelength / doppler_shift

class FCCLattice:
    """Rede cúbica de face centrada de nós STAR."""

    def __init__(self, side_length: float, node_spacing: float):
        self.side_length = side_length
        self.node_spacing = node_spacing
        self.nodes: List[STARNode] = self._generate_lattice()

    def _generate_lattice(self) -> List[STARNode]:
        """Gera posições FCC: (0,0,0), (0,½,½), (½,0,½), (½,½,0) em coordenadas de célula."""
        nodes = []
        cells_per_side = int(self.side_length / self.node_spacing)

        for i in range(cells_per_side):
            for j in range(cells_per_side):
                for k in range(cells_per_side):
                    # Posições base da célula FCC
                    base_positions = [
                        np.array([i, j, k]),
                        np.array([i, j+0.5, k+0.5]),
                        np.array([i+0.5, j, k+0.5]),
                        np.array([i+0.5, j+0.5, k])
                    ]
                    for pos in base_positions:
                        pos_m = pos * self.node_spacing
                        # 'a' inicial uniforme, depois será perturbado por massas
                        nodes.append(STARNode(pos_m, np.random.randn(3), 1.0))
        return nodes

    def apply_gravitational_potential(self, mass: float, position: np.ndarray):
        """
        Ajusta o campo escalar 'a' dos nós baseado no potencial gravitacional.
        Implementa Beckmann: n(r) = 1 + 2GM/rc₀²
        """
        G = 6.67430e-11
        c0 = 299792458.0

        for node in self.nodes:
            r = np.linalg.norm(node.position - position)
            if r > 0:
                # Correção relativística: n = 1 + 2GM/rc²
                node.scalar_a = 1.0 + (2 * G * mass) / (r * c0**2)
