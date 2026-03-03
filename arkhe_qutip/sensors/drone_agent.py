"""
Drone Agent: Arkhe(n) Node implementation for autonomous drones in hyperbolic geometry.
"""

import numpy as np
from qutip import basis, tensor, ptrace
from arkhe_qutip.core.arkhe_qobj import ArkheQobj

class DroneAgentNode(ArkheQobj):
    """
    Autonomous drone as an Arkhe(n) node in H2 (Hyperbolic space).
    """

    def __init__(self, node_id, position, battery=1.0):
        # Base state: operational mode as effective qubit
        # |0⟩ = PATROL, |1⟩ = ACTIVE_DETECTION
        super().__init__(basis(2, 0), node_id=node_id)

        self.pos = position  # (x, y) in Poincaré upper half-plane (y > 0)
        self.battery = battery

        # Embedded THz Sensor
        self.thz = {
            'Fermi': 0.85,  # eV
            'modes': [2.49, 3.90, 6.14],  # THz
            'Q': 58.73,
            'C_sensor': 1 - 1/58.73  # ~0.983
        }

        # Arkhe(n) Metrics
        self.C_local = 0.5
        self.F_local = 0.5
        self.z = 1.0

        # Connectivity and Entanglement
        self.neighbors = []
        self.entangled_fleet = []

        # Cognitive Load (Art. 1 Constitution)
        self.cognitive_load = 0.0

    def hyperbolic_distance(self, other_pos):
        """
        Hyperbolic distance in H2 (upper half-plane):
        d = arcosh(1 + |x1-x2|^2 + |y1-y2|^2 / (2*y1*y2))
        """
        x1, y1 = self.pos
        x2, y2 = other_pos
        arg = 1 + ((x1 - x2)**2 + (y1 - y2)**2) / (2 * y1 * y2)
        # Numerical stability: arccosh is defined for arg >= 1
        return np.arccosh(max(1.0, arg))

    def update_coherence(self, fleet_positions, R_comm=2.0):
        """Update C_local based on connectivity."""
        # Count neighbors within R_comm (hyperbolic distance)
        n_neighbors = sum([
            1 for p in fleet_positions
            if self.hyperbolic_distance(p) < R_comm and not np.array_equal(p, self.pos)
        ])

        # C_local saturates with ~3 neighbors
        self.C_local = 1 - np.exp(-n_neighbors / 3)
        self.coherence = self.C_local # sync with parent attribute

        # Fluctuation = maneuverability (battery + neighborhood density)
        self.F_local = self.battery * (1 - 0.1 * min(n_neighbors, 5))
        # Note: self.fluctuation is read-only and maintained by self.coherence setter

        self.z = self.F_local / (self.C_local + 1e-6)

        return self.C_local

    def detect_thz(self, target_signature, atmospheric_noise=0.1):
        """
        Simulate THz detection with hyperbolic correction.
        """
        # Tune to target signature
        # We assume modes[1] is the primary sensing mode
        detuning = abs(self.thz['modes'][1] - target_signature)
        tuning_quality = 1 / (1 + detuning**2)

        # Altitude factor: higher y = lower signal (atmospheric density/distance)
        altitude_factor = 1 / np.sqrt(self.pos[1])

        # Signal
        signal = self.thz['C_sensor'] * tuning_quality * altitude_factor
        signal += np.random.normal(0, atmospheric_noise)

        # Update cognitive load
        self.cognitive_load += 0.1
        if self.cognitive_load > 0.7:
            # Art. 1: Overload - force safety mode (Patrol)
            self._update_state(basis(2, 0))
            self.cognitive_load = 0.0

        return max(0, signal)

    def entangle_with_fleet(self, fleet):
        """
        Create GHZ entanglement with local fleet.
        """
        if len(fleet) < 2:
            return None

        n = len(fleet)
        # GHZ state: (|0...0> + |1...1>) / sqrt(2)
        # Note: In practice for large n, this is memory intensive
        try:
            ghz = (tensor([basis(2,0)]*n) + tensor([basis(2,1)]*n)).unit()
        except Exception:
            # Fallback if too many qubits
            ghz = None

        for drone in fleet:
            drone.entangled_fleet = [f.node_id for f in fleet if f != drone]
            drone.C_local = 0.5
            drone.coherence = 0.5

        return {
            'C_global': 1.0,
            'C_locals': [0.5]*n,
            'emergence': True,
            'ghz_state': ghz
        }
