import pytest
import numpy as np
from arkhe.wifi_radar import WiFiRadar

def test_pearson_correlation():
    series_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    series_b = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    corr = WiFiRadar.calculate_pearson(series_a, series_b)
    assert corr > 0.95

def test_radar_inference():
    radar = WiFiRadar(node_count=10)
    positions = radar.infer_positions()
    assert len(positions) == 10
    assert "id" in positions[0]
    assert "x" in positions[0]
    # Drone (0) and Demon (1) should have correlation 0.94
    assert positions[1]["correlation_with_source"] == 0.94

def test_radar_summary():
    from arkhe.wifi_radar import get_radar_summary
    summary = get_radar_summary()
    assert summary["Nodes_Detected"] == 42
    assert summary["System"] == "Gemini 3 Deep Think"
