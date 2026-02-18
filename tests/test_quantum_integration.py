import unittest
import numpy as np
from UrbanSkyOS.core.safe_core import SafeCore, QuantumState
from UrbanSkyOS.connectivity.handover import QuantumHandoverProtocol
from UrbanSkyOS.intelligence.quantum_pilot import QuantumPilotCore
from UrbanSkyOS.core.drone_node import DroneNode

class TestQuantumIntegration(unittest.TestCase):
    def test_safe_core_metrics(self):
        safe = SafeCore(n_qubits=4)
        self.assertEqual(safe.coherence, 1.0)
        self.assertEqual(safe.phi, 0.0)

        # Apply gate to change state
        safe.apply_gate(np.eye(16), [0])
        self.assertLess(safe.coherence, 1.0)
        self.assertGreater(safe.phi, 0.0)

    def test_handover_protocol(self):
        safe = SafeCore(n_qubits=4)
        handover = QuantumHandoverProtocol()

        # Initial state amplitudes
        initial_amps = safe.quantum_state.amplitudes.copy()

        # Freeze
        frozen = handover.freeze_quantum_state(safe)
        self.assertIn('amplitudes', frozen)
        self.assertEqual(frozen['hash'], handover._hash_state(initial_amps))

        # Transfer to classical
        density = handover.transfer_to_classical(frozen)
        self.assertEqual(density.shape, (16, 16))

        # Reset safe core state
        safe.quantum_state.amplitudes.fill(0)

        # Resume
        handover.resume_quantum(safe, frozen)
        np.testing.assert_array_almost_equal(safe.quantum_state.amplitudes, initial_amps)

    def test_quantum_pilot(self):
        safe = SafeCore(n_qubits=10)
        handover = QuantumHandoverProtocol()
        pilot = QuantumPilotCore(safe, handover)

        percept = pilot.perceive()
        self.assertEqual(len(percept), 1024)
        self.assertAlmostEqual(np.linalg.norm(percept), 1.0)

        decision = pilot.decide(percept)
        self.assertEqual(len(decision), 1024)

        action = pilot.act(decision)
        self.assertIn('action', action)
        self.assertIn('delta_v', action)
        self.assertEqual(action['delta_v'], 47.56)

    def test_drone_node_quantum_integration(self):
        drone = DroneNode("TEST-DRONE")
        self.assertTrue(drone.safe_core.active)

        # Run one control loop iteration
        cmds = drone.control_loop(dt=0.01)
        self.assertEqual(len(cmds), 4)

        # Manually lower coherence to trigger handover
        drone.safe_core.coherence = 0.5
        drone.control_loop(dt=0.01)
        self.assertTrue(drone.safe_core.handover_mode)

if __name__ == '__main__':
    unittest.main()
