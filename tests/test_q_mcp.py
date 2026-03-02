import unittest
import asyncio
from cosmos.mcp import QM_Context_Protocol, CoherenceMonitor
from cosmos.network import SwarmOrchestrator

class TestQMCP(unittest.TestCase):
    def test_mcp_teleportation(self):
        mcp = QM_Context_Protocol()
        insight = "Test Insight"

        reconstructed = asyncio.run(
            mcp.teleport_context("Code_Swarm", "Hardware_Swarm", insight)
        )

        self.assertTrue(reconstructed.startswith("Ação_Acelerada_"))
        self.assertIn("via", reconstructed)

    def test_swarm_orchestrator(self):
        mcp = QM_Context_Protocol()
        orchestrator = SwarmOrchestrator(mcp)

        self.assertEqual(orchestrator.active_swarms["Code_Swarm"]["agents"], 16)
        orchestrator.scale_agents("Code_Swarm", 1000)
        self.assertEqual(orchestrator.active_swarms["Code_Swarm"]["agents"], 1000)

        metrics = orchestrator.get_acceleration_metrics()
        self.assertEqual(metrics["parallelization_factor"], 1000)
        self.assertEqual(metrics["total_agents"], 1150)

    def test_coherence_monitor(self):
        monitor = CoherenceMonitor()
        self.assertTrue(monitor.check_stability(1000))
        self.assertGreater(monitor.global_coherence, 0.8)
        # Extreme load
        for _ in range(100):
            monitor.check_stability(1000000)
        self.assertFalse(monitor.check_stability(1000000))

if __name__ == "__main__":
    unittest.main()
