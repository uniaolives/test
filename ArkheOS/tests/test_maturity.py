import pytest
from arkhe.attention import AttentionEngine
from arkhe.vacuum_energy import VacuumEngine
from arkhe.wifi_radar import WiFiRadar

def test_active_inference():
    ae = AttentionEngine()
    # High local gradient should increase attention density
    density = ae.calculate_attention_density(0.24)
    assert density > 1.0

    # Precision weighting
    weighted = ae.active_inference_step(0.1)
    assert weighted == pytest.approx(0.1 * (1.0 / 0.15**2))

def test_vacuum_warp():
    ve = VacuumEngine()
    # Energy extraction
    energy = ve.extract_energy()
    assert energy > 0

    # Warp drive
    res = ve.engage_warp_drive((1000, 500, -50))
    assert res["Status"] == "WARP_ENGAGED"
    assert res["G_Force"] == 0

def test_radar_correlation():
    radar = WiFiRadar(node_count=5)
    matrix = radar.get_correlation_matrix()
    # Drone-Demon correlation should be 0.94
    assert matrix[0, 1] == 0.94
