"""
Graphene THz Sensor: Arkhe(n) Node implementation for THz sensing.
Based on graphene metamaterials with tunable Fermi levels.
"""

import numpy as np
from qutip import basis, bell_state, ptrace
from arkhe_qutip.core.arkhe_qobj import ArkheQobj

class GrapheneTHzSensorNode(ArkheQobj):
    """
    Sensor THz as an Arkhe(n) quantum node with plasmonic properties.
    """

    def __init__(self, node_id, fermi_level=0.85):
        # Base state: resonance mode as effective qubit
        # |0⟩ = non-excited mode, |1⟩ = excited mode
        super().__init__(basis(2, 0), node_id=node_id)

        self.fermi_level = fermi_level  # eV
        self.modes = [
            {'f': 2.49, 'Q': 45, 'absorption': 0.719},
            {'f': 3.90, 'Q': 58.73, 'absorption': 0.999},  # optimum mode
            {'f': 6.14, 'Q': 52, 'absorption': 0.996}
        ]

        # Local coherence based on mean Q
        q_mean = np.mean([m['Q'] for m in self.modes])
        self._coherence = 1 - 1/q_mean  # ≈ 0.98
        self._fluctuation = 1.0 - self._coherence

        # Tuning capacity = fluctuation
        self.tuning_range = 0.3  # eV (0.7 to 1.0)

        # Entanglement state
        self.entangled_peers = []
        self.bell_pair = None  # entangled state with peer

    def tune(self, voltage_ratio):
        """Tune via Fermi level (meta-handover)."""
        self.fermi_level = 0.7 + 0.3 * np.sqrt(voltage_ratio)

        # Blueshift: f ∝ sqrt(E_F)
        for mode in self.modes:
            if 'f_base' not in mode:
                mode['f_base'] = mode['f']
            mode['f'] = mode['f_base'] * np.sqrt(self.fermi_level / 0.7)

        # Update coherence (tuning slightly reduces Q)
        self.update_coherence(-0.02 * voltage_ratio)

        return self.modes[1]['f']  # return frequency of optimum mode

    def absorb(self, frequency, analyte_n=1.0):
        """
        Simulate absorption handover with analyte shift.
        """
        # Resonance shift: Δf/f = α(n-1)
        alpha = 0.1  # simplified sensitivity

        responses = []
        for mode in self.modes:
            f_shifted = mode['f'] * (1 + alpha * (analyte_n - 1))
            # Lorentzian
            res = mode['absorption'] / (1 + ((frequency - f_shifted)/(mode['f']/mode['Q']))**2)
            responses.append(res)

        # Measurement coherence
        measurement_coherence = max(responses)  # highest peak

        # Record handover
        self.record_handover(
            handover_id=f"absorb_{np.random.randint(10000)}",
            target_id="environment",
            metadata={
                'frequency': frequency,
                'analyte_n': analyte_n,
                'responses': responses,
                'max_response': measurement_coherence
            }
        )

        return responses, measurement_coherence

    def entangle_with(self, peer):
        """
        Create Bell entanglement with another sensor (non-local handover).
        """
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell = bell_state('00')

        # Assign half of the state to each sensor
        self.bell_pair = bell
        peer.bell_pair = bell

        # Correctly update peers
        if peer.node_id not in self.entangled_peers:
            self.entangled_peers.append(peer.node_id)
        if self.node_id not in peer.entangled_peers:
            peer.entangled_peers.append(self.node_id)

        # Local coherences are reduced (maximal mixture)
        c_local = 0.5
        c_global = 1.0

        self.coherence = c_local
        peer.coherence = c_local

        return {
            'C_global': c_global,
            'C_local_self': c_local,
            'C_local_peer': c_local,
            'emergence': c_global > max(c_local, c_local),
            'violation_bell': True
        }
