# tests/test_astrodynamics.py
import pytest
from arkhe.astrodynamics import OrbitalObservatory, get_default_catalog
from arkhe.geodesic import ArkheSatellite, WhippleShield

def test_orbital_selectivity():
    obs = OrbitalObservatory(handovers=9045)
    catalog = get_default_catalog()
    for sat in catalog:
        obs.add_satellite(sat)

    fraction = obs.calculate_active_fraction()
    # (6 satellites + 6 Hansson Handels) / 9045 = 12 / 9045 ≈ 0.0013267
    assert pytest.approx(fraction, abs=1e-6) == 12 / 9045

    ratio = obs.get_selectivity_ratio()
    # 0.005 / (12/9045) ≈ 3.76875
    assert pytest.approx(ratio, abs=0.1) == 3.8

def test_whipple_shield():
    shield = WhippleShield(remaining_lifetime_s=999.819, competence_h=6)

    # Impact under capacity
    result = shield.assess_impact(1.0)
    assert "CONTAINED" in result
    assert "16.7%" in result

    # Impact over capacity
    result = shield.assess_impact(7.0)
    assert "CRITICAL" in result
