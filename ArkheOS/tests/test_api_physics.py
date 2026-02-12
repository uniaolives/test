
import pytest
from arkhe.physics import QuantumGravityEngine
from arkhe.api import ArkheAPI, ContractIntegrity

def test_graviton_mass():
    m_grav = QuantumGravityEngine.calculate_graviton_mass()
    assert 5.0e-53 < m_grav < 6.0e-53

def test_api_handle_request():
    api = ArkheAPI()
    resp = api.handle_request("GET", "/coherence", {})
    assert resp["status"] == 200
    assert resp["body"]["C"] == 0.86
    assert "Arkhe-Phi-Inst" in resp["headers"]

def test_api_entanglement():
    api = ArkheAPI()
    resp = api.handle_request("POST", "/entangle", {}, {"omega": 0.07})
    assert resp["status"] == 201
    session_id = resp["body"]["session_id"]

    # Test request with session
    resp2 = api.handle_request("GET", "/ω/0.07/dvm1.cavity", {"Arkhe-Entanglement": session_id})
    assert "déjà vu" in resp2["body"]

def test_contract_integrity_reentry(capsys):
    ContractIntegrity._counts = {} # Reset
    ContractIntegrity.detect_spec_reentry(9050)
    ContractIntegrity.detect_spec_reentry(9050)
    captured = capsys.readouterr()
    assert "integrada" in captured.out
    assert "detectado" in captured.out
