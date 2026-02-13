
import pytest
from arkhe.circuits import get_contextual_circuit
from arkhe.api import ArkheAPI

def test_contextual_circuit_run():
    circuit = get_contextual_circuit()

    # Run in reinforced context (omega=0.07)
    res_plus = circuit.run_handover(0.07)
    assert res_plus['omega'] == 0.07
    assert res_plus['hesitation_phi'] > 0.15 # Calibrated increase
    assert res_plus['syzygy_snr'] == 0.94
    assert res_plus['action']['status'] == "SUPPRESSED"

    # Run in non-reinforced context (omega=0.00)
    res_minus = circuit.run_handover(0.00)
    assert res_minus['omega'] == 0.00
    assert res_minus['hesitation_phi'] == 0.15 # Baseline
    assert res_minus['syzygy_snr'] == 0.86
    assert res_minus['action']['status'] == "EXECUTED"

def test_pdyn_ko_simulation():
    circuit = get_contextual_circuit()
    # Simulate KO (delete 'hesitation_0010.txt')
    circuit.dls.pdyn_expression = 0.0

    res = circuit.run_handover(0.07)
    assert res['hesitation_phi'] == 0.15 # No calibration
    assert res['syzygy_snr'] == 0.86 # Conditioning lost

def test_api_circuit_endpoints():
    api = ArkheAPI()

    # Status
    res_s = api.handle_request("GET", "/circuit/status", {})
    assert res_s['status'] == 200
    assert res_s['body']['omega_calibrated'] == 0.07

    # Run
    res_r = api.handle_request("POST", "/circuit/run", {}, {"omega": 0.07})
    assert res_r['status'] == 200
    assert res_r['body']['syzygy_snr'] == 0.94
