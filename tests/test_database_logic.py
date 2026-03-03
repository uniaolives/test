#!/usr/bin/env python3
# tests/test_database_logic.py
import sys
import os
import unittest

# Add metalanguage to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.arkhe_database import (
    SQLCognitiveLayer,
    MongoDBCognitiveLayer,
    RedisCognitiveLayer,
    DatabaseConservationGuard,
    PHI
)

class TestDatabaseLogic(unittest.TestCase):
    def setUp(self):
        self.guard = DatabaseConservationGuard()

    def test_sql_regime(self):
        layer = SQLCognitiveLayer()
        metrics = layer.get_metrics()
        self.assertEqual(metrics.regime, "DETERMINISTIC")
        self.assertTrue(metrics.z < PHI)
        self.assertTrue(self.guard.verify(metrics))

    def test_mongodb_regime(self):
        layer = MongoDBCognitiveLayer()
        metrics = layer.get_metrics()
        self.assertEqual(metrics.regime, "CRITICAL")
        self.assertAlmostEqual(metrics.z, PHI)
        self.assertTrue(self.guard.verify(metrics))

    def test_redis_regime(self):
        layer = RedisCognitiveLayer()
        metrics = layer.get_metrics()
        self.assertEqual(metrics.regime, "STOCHASTIC")
        self.assertTrue(metrics.z > PHI)
        self.assertTrue(self.guard.verify(metrics))

if __name__ == "__main__":
    unittest.main()
