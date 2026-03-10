import unittest
from chronos import Client, Transaction, ConsistencyLevel

class TestChronosSDK(unittest.TestCase):
    def setUp(self):
        self.client = Client(api_key="ck_test_123")

    def test_client_init(self):
        self.assertEqual(self.client.api_key, "ck_test_123")
        self.assertEqual(self.client.region, "us-east-1")

    def test_transaction_begin(self):
        tx = self.client.begin_transaction()
        self.assertIsInstance(tx, Transaction)
        self.assertTrue(tx.tx_id.startswith("orb_"))

    def test_record_event(self):
        tx = self.client.begin_transaction()
        tx.record_event("test_event")
        self.assertEqual(len(tx.events), 1)
        self.assertEqual(tx.events[0]["event"], "test_event")

    def test_commit(self):
        tx = self.client.begin_transaction()
        committed_time = tx.commit()
        self.assertIsInstance(committed_time, float)

    def test_cluster_coherence(self):
        coherence = self.client.get_cluster_coherence()
        self.assertGreater(coherence, 0.9)
        self.assertLessEqual(coherence, 1.0)

if __name__ == "__main__":
    unittest.main()
