
import pytest
from arkhe.abiogenesis import get_abiogenesis
from arkhe.api import ArkheAPI

def test_eigen_threshold():
    engine = get_abiogenesis()
    res = engine.calculate_eigen_threshold(45, 0.941)
    # 1 / (1 - 0.941) = 16.949...
    assert res > 16.9
    assert res < 17.0

def test_selection_simulation():
    engine = get_abiogenesis()
    res = engine.run_selection_simulation(100)

    assert res['status'] == "SELECTION_CYCLE_COMPLETE"
    assert res['cycles'] == 100
    assert res['emergent_variant']['name'] == "QT45-V3"
    assert res['emergent_variant']['size'] == 47
    assert res['ledger_block'] == 9082

def test_parallel_coupling_h7():
    engine = get_abiogenesis()
    res = engine.parallel_coupling("H7")

    assert res['identity'] == "QT45 IS H7"
    assert res['satoshi'] == 7.27

def test_api_abiogenesis_endpoints():
    api = ArkheAPI()

    # Simulate
    res_sim = api.handle_request("POST", "/abiogenesis/simulate", {}, {"cycles": 100})
    assert res_sim['status'] == 200
    assert res_sim['body']['status'] == "SELECTION_CYCLE_COMPLETE"

    # Parallel
    res_par = api.handle_request("POST", "/abiogenesis/parallel", {}, {"block": "H7"})
    assert res_par['status'] == 200
    assert res_par['body']['identity'] == "QT45 IS H7"
