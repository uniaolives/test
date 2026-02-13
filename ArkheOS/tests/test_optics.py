
import pytest
from arkhe.optics import get_jarvis_sensor, get_connectome
from arkhe.api import ArkheAPI

def test_jarvis_sensor_illumination():
    sensor = get_jarvis_sensor()

    # Test scanless (optimal)
    res_scanless = sensor.apply_illumination("scanless", 0.63)
    assert res_scanless['snr'] == 94.0
    assert res_scanless['kinetics'] == "Fast (optimal)"

    # Test scanning (saturated)
    res_scanning = sensor.apply_illumination("scanning", 50.0)
    assert res_scanning['snr'] < 1.0
    assert res_scanning['kinetics'] == "Slow (saturated)"

def test_jarvis_ap_detection():
    sensor = get_jarvis_sensor()
    sensor.apply_illumination("scanless", 0.63)

    # AP at 0.73 rad threshold
    res_ap = sensor.detect_action_potential(0.74)
    assert res_ap['detected'] == True
    assert res_ap['snr'] == 94.0

def test_functional_connectome():
    conn = get_connectome()
    res = conn.map_connectivity()

    assert res['status'] == "FUNCTIONAL_CONNECTOME_MAPPED"
    assert res['mean_correlation'] == 0.94
    assert res['ledger_block'] == 9083

def test_api_optics_endpoints():
    api = ArkheAPI()

    # Illumination
    res_ill = api.handle_request("POST", "/optics/jarvis/illumination", {}, {"mode": "scanless", "irradiance": 0.63})
    assert res_ill['status'] == 200
    assert res_ill['body']['snr'] == 94.0

    # Detect
    res_det = api.handle_request("POST", "/optics/jarvis/detect", {}, {"voltage": 0.75})
    assert res_det['status'] == 200
    assert res_det['body']['detected'] == True

    # Connectome
    res_conn = api.handle_request("POST", "/optics/jarvis/connectome", {}, {})
    assert res_conn['status'] == 200
    assert res_conn['body']['mean_correlation'] == 0.94
