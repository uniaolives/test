import pytest
import numpy as np
from arkhe.deep_belief import DeepBeliefNetwork, KalmanFilterArkhe
from arkhe.feedback_economy import FeedbackEconomy, Echo2Arkhe
from arkhe.resilience import PerceptualResilience, BlindSpotSimulator

def test_kalman_filter_convergence():
    kf = KalmanFilterArkhe()
    measurement = 0.98
    for _ in range(10):
        filtered = kf.update(measurement)
    assert 0.94 <= filtered <= 1.0

def test_echo2_distributed_reward():
    engine = Echo2Arkhe(satoshi=7.27)
    node_id = "seoul_01"
    engine.add_node(node_id, 0.03)

    initial_satoshi = engine.global_satoshi
    engine.async_rollout(node_id, "cmd", 0.94) # Added missing command arg

    assert engine.global_satoshi > initial_satoshi
    assert engine.nodes[node_id]['accumulated_reward'] == 0.94

def test_blind_spot_reconstruction():
    resilience = PerceptualResilience()
    # No input (blind spot)
    c, syz = resilience.enforce_global_constraints(local_input=None)
    assert syz == 0.9402 # Updated for Γ_∞+51
    assert c == 0.86

def test_resilience_stress_test():
    simulator = BlindSpotSimulator()
    simulator.inject_blind_spot(0.03, 10)
    assert simulator.run_stress_test(50) is True

def test_feedback_report():
    from arkhe.feedback_economy import get_feedback_report
    report = get_feedback_report()
    assert report['State'] == "Γ_∞+46"
    assert "Echo-2" in report['Infrastructure'] # Fixed typo

def test_mathematical_report():
    from arkhe.deep_belief import get_mathematical_framework_report
    report = get_mathematical_framework_report()
    assert report['State'] == "Γ_∞+44"
    assert "Kalman" in report['Filtering'] # Fixed key (was Kalman_Filter in test, but Filtering in code)
