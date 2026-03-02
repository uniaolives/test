
import pytest
import numpy as np
from arkhe.hematology import HematologyEngine, ScarElastography
from arkhe.sigma_model import SigmaModelEngine, SigmaModelParameters
from arkhe.orch_or import OrchOREngine

def test_hematology_coagulation():
    result = HematologyEngine.run_cascade()
    assert result.fator_vii > 0.8
    assert result.fibrina > 0.5
    assert result.risco_trombo_pct < 1.0
    assert result.inr == 1.02

def test_scar_mapping():
    scar_map = ScarElastography.get_full_map()
    assert len(scar_map) == 8 # 7 nodes + vacuum
    assert scar_map['QN-07']['pressure'] > scar_map['WP1']['pressure']
    assert scar_map['Vácuo WP1']['density'] < scar_map['WP1']['density']

def test_sigma_model_fixed_point():
    params = SigmaModelParameters()
    report = SigmaModelEngine.get_effective_action_report(params)
    assert report["Status"] == "FIXED_POINT (β=0)"
    assert "7.27 bits" in report["Action (S)"]
    assert SigmaModelEngine.check_fixed_point_condition(0, 0, 0) is True

def test_orch_or_conscience():
    # Test Penrose Tau for Kernel (omega=0.12)
    tau = OrchOREngine.calculate_penrose_tau(0.12)
    # 1.0 / (0.73 * 7.27 * 0.12) * 11.4 * 100 approx 1790
    assert 1700 < tau < 1850

    # Test EEG mapping
    assert "Gama" in OrchOREngine.get_eeg_mapping(0.12)
    assert "Delta" in OrchOREngine.get_eeg_mapping(0.00)
    assert "Gama alto" in OrchOREngine.get_eeg_mapping(0.21)

def test_penrose_tau_range():
    # Test that tau decreases as omega (energy gap) increases
    tau_low = OrchOREngine.calculate_penrose_tau(0.03)
    tau_high = OrchOREngine.calculate_penrose_tau(0.21)
    assert tau_high < tau_low
    assert OrchOREngine.calculate_penrose_tau(0) == float('inf')
