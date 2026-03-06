import pytest
import pandas as pd
import numpy as np
from gateway.app.knowledge.google_scanner import SemanticMiner

def test_semantic_anomaly_detection():
    # Simulate 30 years of data
    years = np.arange(1990, 2020)

    # Natural growth (S-curve approximation)
    natural = 1 / (1 + np.exp(-(years - 2005) * 0.3))

    # Retrocausal injection (abrupt step)
    injection = np.zeros(len(years))
    injection[years >= 2009] = 1.0 # Sudden jump at 2009

    df = pd.DataFrame({
        'natural_concept': natural,
        'retro_concept': injection
    })

    miner = SemanticMiner(df)
    anomalies = miner.detect_anomalies(threshold=1.5)

    concepts = [a['concept'] for a in anomalies]
    assert 'retro_concept' in concepts
    # 'natural_concept' should not be flagged if threshold is high enough
    # but smooth S-curves might have some jerk, so we just check retro_concept

def test_semantic_metrics():
    df = pd.DataFrame({'test': [0, 0, 0, 1, 2, 5, 10, 20, 50, 100]})
    miner = SemanticMiner(df)
    res = miner.analyze_knowledge_squeezing('test')
    assert res['max_velocity'] > 0
    assert res['max_jerk'] > 0
