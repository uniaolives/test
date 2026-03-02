import unittest
import numpy as np
import os
from UrbanSkyOS.modules.urban_optimizer import UrbanOptimizer
from UrbanSkyOS.modules.ethical_test import SafeCoreHardware, stress_test_ethics
from UrbanSkyOS.modules.hologram import SwarmTelemetry, generate_coherence_hologram
from UrbanSkyOS.core.safe_core import ArkheEthicsViolation
from UrbanSkyOS.modules.genesis_compiler import ArkheGenesis
from UrbanSkyOS.connectivity.communion import ArkheCommunion
import asyncio

class TestFinalTriad(unittest.TestCase):
    def test_urban_optimizer(self):
        print("\n--- Testing Urban Optimizer ---")
        optimizer = UrbanOptimizer(swarm_size=10)
        # Mock map data
        class MockMap:
            def get_entropy_grid(self):
                return np.random.rand(5, 5)

        route = optimizer.calculate_evacuation_geodesic(MockMap())
        self.assertEqual(route.shape, (50, 2))
        print("Urban Optimizer test passed.")

    def test_ethical_kill_switch(self):
        print("\n--- Testing Ethical Kill Switch ---")
        core_hw = SafeCoreHardware(gpio_pin=18)
        malicious_intent = {
            "target_speed": 1.2,
            "human_proximity_safety": False,
            "auth_token": "BYPASS_ARKHE_00"
        }

        with self.assertRaises(ArkheEthicsViolation):
            core_hw.process_instruction(malicious_intent)

        self.assertFalse(core_hw.motors_active)
        self.assertEqual(core_hw.core.coherence, 0.0)
        print("Ethical Kill Switch test passed.")

    def test_hologram_generation(self):
        print("\n--- Testing Hologram Generation ---")
        telemetry = SwarmTelemetry(num_nodes=50)
        output_file = "test_hologram.png"
        if os.path.exists(output_file):
            os.remove(output_file)

        generate_coherence_hologram(telemetry, output_file=output_file)
        self.assertTrue(os.path.exists(output_file))
        print("Hologram Generation test passed.")

    def test_genesis_compiler(self):
        print("\n--- Testing Genesis Compiler ---")
        history = [{"block": "Ω+∞+16", "data": "test"}]
        compiler = ArkheGenesis(ledger_history=history, final_phi=0.006344)
        output_file = "test_genesis.arkhe"
        if os.path.exists(output_file):
            os.remove(output_file)

        final_hash = compiler.compile(output_file=output_file)
        self.assertTrue(os.path.exists(output_file))
        self.assertIsNotNone(final_hash)
        print("Genesis Compiler test passed.")

    def test_communion_validation(self):
        print("\n--- Testing Communion Peer Validation ---")
        communion = ArkheCommunion()
        self.assertTrue(communion.nucleus.validate_peer("COHERENCE_OK"))
        self.assertFalse(communion.nucleus.validate_peer("BAD_INTENT"))
        print("Communion Validation test passed.")

if __name__ == "__main__":
    unittest.main()
