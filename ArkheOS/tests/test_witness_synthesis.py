import pytest
import numpy as np
from arkhe.deep_belief import KalmanFilterArkhe, DeepBeliefNetwork
from arkhe.feedback_economy import FeedbackEconomy, Echo2Arkhe
from arkhe.resilience import BlindSpotCorrespondence, ResilienceEngine

def test_kalman_filter():
    kf = KalmanFilterArkhe()
    res = kf.update(0.98)
    assert 0.94 <= res <= 1.0

def test_feedback_economy():
    fe = FeedbackEconomy(satoshi=7.27)
    engine = fe.engine
    engine.add_node("node1", 0.03)
    engine.async_rollout("node1", "cmd", 0.94)
    assert engine.global_satoshi > 7.27
    assert engine.nodes["node1"]["accumulated_reward"] == 0.94

def test_resilience_blind_spot():
    engine = ResilienceEngine()
    corr = engine.correspondence
    res = corr.test_resilience(0.03, 5)
    assert res['reconstruction_quality'] == 1.0
    assert len(res['syzygy_during_gap']) == 5
    assert res['syzygy_after'] == 0.9402 # Updated for Γ_∞+51

def test_report_states():
    from arkhe.deep_belief import get_mathematical_framework_report
    from arkhe.feedback_economy import get_feedback_report
    from arkhe.resilience import get_resilience_report

    math_rep = get_mathematical_framework_report()
    assert math_rep['State'] == "Γ_∞+44"

    fb_rep = get_feedback_report()
    assert fb_rep['State'] == "Γ_∞+46"

    res_rep = get_resilience_report()
    assert res_rep['State'] == "Γ_∞+51" # Updated to match code
