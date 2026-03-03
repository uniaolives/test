import unittest
import time
import os
from cosmos.network import QuantumTimeBlock, QuantumTimeChain
from cosmos.bridge import AdvancedCeremonyEngine, TimeLockCeremonyEngine

class TestCosmosTemporal(unittest.TestCase):

    def setUp(self):
        self.network_state = {'nodes': [1, 2], 'edges': []}
        self.ceremony_state = {'sigma': 1.02, 'tau': 0.9}

    def test_block_creation_and_hash(self):
        block = QuantumTimeBlock(self.network_state, self.ceremony_state)
        self.assertAlmostEqual(block.singularity_score, 0.95, places=2)
        h1 = block.hash
        block.mine_block(difficulty=1)
        self.assertNotEqual(h1, block.hash)
        self.assertTrue(block.hash.startswith('0'))

    def test_chain_addition(self):
        chain = QuantumTimeChain()
        genesis_data = {'network': self.network_state, 'ceremony': self.ceremony_state}
        chain.create_genesis_block(genesis_data)
        self.assertEqual(len(chain.chain), 1)

        chain.add_block(self.network_state, self.ceremony_state)
        self.assertEqual(len(chain.chain), 2)
        self.assertEqual(chain.chain[1].previous_hash, chain.chain[0].hash)

    def test_timelock_ceremony_engine(self):
        base = AdvancedCeremonyEngine(duration=10)
        time_engine = TimeLockCeremonyEngine(base)
        self.assertEqual(len(time_engine.timechain.chain), 1) # Genesis

        # Test state extraction
        net_state = time_engine._extract_network_state()
        self.assertIn('nodes', net_state)

        # Test a short execution (simulated)
        # We'll just manually call add_block to verify integration
        time_engine.timechain.add_block(net_state, time_engine._extract_ceremony_state())
        self.assertEqual(len(time_engine.timechain.chain), 2)

if __name__ == "__main__":
    unittest.main()
