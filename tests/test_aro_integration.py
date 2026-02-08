import os
import pytest
from src.avalon.governance.aro import AROBridge
from src.avalon.governance.oracle import ScientificOracle

@pytest.fixture(autouse=True)
def cleanup_state():
    state_file = ".test_aro_state.json"
    if os.path.exists(state_file):
        os.remove(state_file)
    yield
    if os.path.exists(state_file):
        os.remove(state_file)

def test_aro_convergence_success():
    aro = AROBridge(state_file=".test_aro_state.json")
    oracle = ScientificOracle()

    # 1. Simulate DAO consensus
    aro.update_dao_consensus(85)

    # 2. Simulate Tech Readiness via Oracle
    readiness = oracle.calculate_readiness_index()
    # Ensure it's high enough for the test
    aro.update_tech_readiness(max(readiness, 95))

    # 3. Set Genomic Proof Fidelity
    proof = "0xdeadbeef"
    aro.set_genomic_fidelity(proof, 100)

    # 4. Initiate Resurrection
    res = aro.initiate_resurrection(proof)

    assert res["status"] == "SUCCESS"
    assert aro.reanimation_active is True

def test_aro_failure_due_to_consensus():
    aro = AROBridge(state_file=".test_aro_state.json")
    aro.update_dao_consensus(50) # Low consensus
    aro.update_tech_readiness(100)

    proof = "0x123"
    aro.set_genomic_fidelity(proof, 100)

    res = aro.initiate_resurrection(proof)
    assert res["status"] == "FAILED"
    assert "DAO consensus low" in res["reason"]

def test_aro_failure_due_to_tech():
    aro = AROBridge(state_file=".test_aro_state.json")
    aro.update_dao_consensus(100)
    aro.update_tech_readiness(40) # Tech not ready

    proof = "0x123"
    aro.set_genomic_fidelity(proof, 100)

    res = aro.initiate_resurrection(proof)
    assert res["status"] == "FAILED"
    assert "Technology not ready" in res["reason"]

def test_aro_failure_due_to_fidelity():
    aro = AROBridge(state_file=".test_aro_state.json")
    aro.update_dao_consensus(100)
    aro.update_tech_readiness(100)

    proof = "0xbad"
    aro.set_genomic_fidelity(proof, 20) # Low fidelity

    res = aro.initiate_resurrection(proof)
    assert res["status"] == "FAILED"
    assert "Genomic fidelity low" in res["reason"]
