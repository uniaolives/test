import pytest
import os
from src.avalon.governance.aro import AROBridge

@pytest.fixture
def aro():
    state_file = ".genesis_test_state.json"
    if os.path.exists(state_file):
        os.remove(state_file)
    bridge = AROBridge(state_file=state_file)
    yield bridge
    if os.path.exists(state_file):
        os.remove(state_file)

def test_genesis_initialization(aro):
    # Before genesis
    status = aro.get_status()
    assert status["verifier_count"] == 0
    assert status["total_reputation"] == 0.0

    # Run genesis
    aro.initialize_genesis()

    status = aro.get_status()
    assert status["verifier_count"] == 21
    # 2100 + 1000*2 + 500*2 + 400*2 + 300*3 + 250*2 + 200*3 + 150*3 + 100*3 = 8650
    assert status["total_reputation"] == 8650.0

    # Verify specific verifier
    assert aro.verifiers["0x716aD3C33A9B9a0A18967357969b94EE7d2ABC10"] == 2100.0 # Satoshi
    assert aro.verifiers["0x9A0B1C2D3E4F56789A0B1C2D3E4F56789A0B1C2D3E4"] == 100.0 # Shafi Goldwasser

def test_dao_consensus_after_genesis(aro):
    aro.initialize_genesis()
    assert aro.dao_consensus == 75
