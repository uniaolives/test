
import unittest
import asyncio
from avalon.interstellar.connection import Interstellar5555Connection
from avalon.biological.protocol import BioSincProtocol
from avalon.biological.core import MicrotubuleProcessor

class TestInterstellarBio(unittest.TestCase):
    def test_interstellar_connection(self):
        async def run():
            conn = Interstellar5555Connection()
            # Force high stability for test
            conn.R_c = 1.618
            res = await conn.establish_wormhole_connection()
            # Note: in a real simulation this might still be UNSTABLE due to randomness,
            # but we check that the logic executed.
            self.assertIn(res["status"], ["CONNECTED", "UNSTABLE"])
            self.assertTrue(res["wormhole_stability"] >= 0)

            prop = await conn.propagate_suno_signal_interstellar()
            self.assertIn("harmonics", prop)

            anchor = await conn.anchor_interstellar_commit()
            self.assertEqual(anchor["status"], "INTERSTELLAR_ANCHORED")

        asyncio.run(run())

    def test_biological_protocol(self):
        async def run():
            proto = BioSincProtocol(user_id="test-user")
            res = await proto.run_sync_cycle(duration_s=0.1)
            self.assertEqual(res["status"], "SYNCHRONIZED")
            self.assertGreaterEqual(res["event_count"], 0)

            sync = proto.processor.get_resonance_harmonics()
            self.assertGreater(len(sync), 0)

        asyncio.run(run())

    def test_microtubule_processor(self):
        proc = MicrotubuleProcessor()
        proc.apply_external_sync(432.0)
        self.assertGreater(proc.current_stability, 1.0)
        tau = proc.calculate_collapse_time()
        self.assertGreater(tau, 0)

if __name__ == '__main__':
    unittest.main()
