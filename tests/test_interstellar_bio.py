
import unittest
import asyncio
from avalon.interstellar.connection import Interstellar5555Connection
from avalon.biological.protocol import BioSincProtocol
from avalon.biological.core import MicrotubuleQuantumCore
from avalon.biological.holography import MicrotubuleHolographicField

class TestInterstellarBio(unittest.TestCase):
    def test_interstellar_connection(self):
        async def run():
            conn = Interstellar5555Connection()
            # Force high stability for test
            conn.R_c = 1.618
            res = await conn.establish_wormhole_connection()
            self.assertIn(res["status"], ["CONNECTED", "UNSTABLE"])
            self.assertTrue(res["wormhole_stability"] > 0)

            prop = await conn.propagate_suno_signal_interstellar()
            self.assertIn("harmonics", prop)

            anchor = await conn.anchor_interstellar_commit()
            self.assertEqual(anchor["status"], "INTERSTELLAR_ANCHORED")

        asyncio.run(run())

    def test_biological_protocol(self):
        async def run():
            proto = BioSincProtocol(user_id="test-user")
            res = await proto.induce_resonance(40.0)
            self.assertTrue(res["resonance_induced"])
            self.assertEqual(res["target_frequency"], 40.0)

            sync = await proto.synchronize_with_interstellar()
            self.assertEqual(sync["synchronization"], "ESTABLISHED")

            encode = await proto.encode_holographic_memory(b"test data")
            self.assertTrue(encode["encoding_successful"])

        asyncio.run(run())

    def test_microtubule_core(self):
        core = MicrotubuleQuantumCore()
        freqs = core.calculate_resonance_frequencies()
        self.assertIn("critical_resonance", freqs)
        self.assertTrue(freqs["critical_resonance"] > 1e12)

if __name__ == '__main__':
    unittest.main()
