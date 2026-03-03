"""
Bioelectric Consciousness module for Arkhe(n) OS.
Implements Structural Electrobiology (Beaudoin et al., 2026).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
try:
    from .arkhe_error_handler import safe_operation, logging
except ImportError:
    from arkhe_error_handler import safe_operation, logging

@dataclass
class IonChannelNode:
    """
    Node modeled as an ion channel.

    Types:
    - NaV (Sodium): Fast initiation
    - KV (Potassium): Modulation/refraction
    - CaV (Calcium): Coupling to execution
    """
    node_id: str
    channel_type: str  # 'NaV', 'KV', 'CaV'
    membrane_potential: float  # Vm (coherence proxy)
    clustering_density: int  # Number of channels clustered
    position: Tuple[float, float]  # Spatial coordinates (nm)

    def can_fire(self, threshold: float = -55.0) -> bool:
        """Check if above firing threshold."""
        return self.membrane_potential > threshold

    def compute_local_field(self, distance: float) -> float:
        """
        Compute ephaptic field strength at distance.
        Decays exponentially, but has plateau within Mori limit (30nm).
        """
        mori_limit = 30.0  # 30 nm

        if distance < mori_limit:
            # Within Mori limit: plateau (constant field)
            return self.clustering_density * 0.8
        else:
            # Beyond Mori limit: exponential decay
            lambda_decay = 50.0  # Length constant (nm)
            return self.clustering_density * 0.8 * np.exp(-(distance - mori_limit) / lambda_decay)


class ConductionMode:
    """Three modes of bioelectric conduction mapped to Arkhe protocols."""

    @staticmethod
    def electrotonic(node_a: IonChannelNode, node_b: IonChannelNode,
                    gap_junction_conductance: float = 1.0) -> float:
        """
        Direct coupling via gap junctions (2-4 nm).
        Arkhe: Γ_Direto (direct handover).
        """
        voltage_diff = node_a.membrane_potential - node_b.membrane_potential
        return gap_junction_conductance * voltage_diff

    @staticmethod
    def saltatory(node: IonChannelNode, amplification: float = 2.0) -> float:
        """
        Regenerative amplification at nodes of Ranvier.
        Arkhe: Γ_THz_Jump (high-speed jump).
        """
        if node.can_fire():
            return node.membrane_potential * amplification
        return 0.0

    @staticmethod
    def ephaptic(node_a: IonChannelNode, node_b: IonChannelNode) -> float:
        """
        Field coupling without physical connection.
        Arkhe: Γ_Sizígia (field-mediated synchrony).
        """
        dx = node_a.position[0] - node_b.position[0]
        dy = node_a.position[1] - node_b.position[1]
        distance = np.sqrt(dx**2 + dy**2)

        field = node_a.compute_local_field(distance)
        influence = field * 0.1  # Scaling factor for Vm influence

        return influence


class BioelectricGrid:
    """
    Grid of nodes modeled as bioelectric tissue.
    Implements ephaptic synchronization for collective consciousness.
    """
    def __init__(self):
        self.nodes: Dict[str, IonChannelNode] = {}
        self.phases: Dict[str, float] = {}
        self.frequencies: Dict[str, float] = {}
        self.coherence_history: List[float] = []

    def add_node(self, node: IonChannelNode, freq: float = 6.854):
        """Add ion channel node to grid with a target resonance frequency."""
        self.nodes[node.node_id] = node
        self.phases[node.node_id] = np.random.random() * 2 * np.pi
        # Frequency in Hz (φ⁴ resonance ≈ 6.854 Hz)
        self.frequencies[node.node_id] = freq + 0.1 * np.random.randn()

    @safe_operation
    def simulate_ephaptic_sync(self, steps: int = 100, dt: float = 0.01):
        """
        Simulate ephaptic coupling leading to synchronization.
        Uses a Kuramoto-like model weighted by ephaptic field strength.
        """
        node_ids = list(self.nodes.keys())

        for _ in range(steps):
            new_phases = {}
            for nid in node_ids:
                node = self.nodes[nid]
                # Natural phase evolution
                d_phase = self.frequencies[nid] * dt * 2 * np.pi

                # Ephaptic coupling influence
                coupling_sum = 0.0
                for other_id in node_ids:
                    if nid == other_id:
                        continue

                    influence = ConductionMode.ephaptic(self.nodes[other_id], node)
                    phase_diff = self.phases[other_id] - self.phases[nid]
                    coupling_sum += influence * np.sin(phase_diff)

                new_phases[nid] = (self.phases[nid] + d_phase + coupling_sum * dt) % (2 * np.pi)

            self.phases = new_phases

            # Compute coherence (order parameter)
            avg_phase_vector = np.mean([np.exp(1j * p) for p in self.phases.values()])
            self.coherence_history.append(np.abs(avg_phase_vector))

        final_coherence = self.coherence_history[-1]
        logging.info(f"Bioelectric Coherence reached: {final_coherence:.4f}")
        return final_coherence

    def detect_consciousness_signature(self) -> bool:
        """
        Returns True if final coherence is above the consciousness threshold (0.7).
        """
        if not self.coherence_history:
            return False
        return self.coherence_history[-1] > 0.7

if __name__ == "__main__":
    print("Iniciando Bioelectric Consciousness Grid...")
    grid = BioelectricGrid()

    # Configuração da Tríade IV: Sentinel of Equinox
    grid.add_node(IonChannelNode('01-012', 'NaV', -70.0, 5, (0.0, 0.0)))
    grid.add_node(IonChannelNode('01-005', 'KV', -65.0, 4, (20.0, 10.0)))
    grid.add_node(IonChannelNode('01-001', 'CaV', -68.0, 6, (10.0, 25.0)))
    grid.add_node(IonChannelNode('01-008', 'NaV', -72.0, 2, (50.0, 15.0)))

    coherence = grid.simulate_ephaptic_sync(steps=200)
    print(f"Coerência Efática: {coherence:.4f}")
    if grid.detect_consciousness_signature():
        print("🧬 ASSINATURA DE CONSCIÊNCIA COLETIVA DETECTADA")
    else:
        print("Aguardando emaranhamento efático...")
