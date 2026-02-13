
import pytest
from arkhe.coupling_language import get_coupling_interpreter
from arkhe.api import ArkheAPI

def test_coupling_interpreter_resolve():
    interpreter = get_coupling_interpreter()
    res = interpreter.resolve_prompt("What is the curvature?")

    assert res['status'] == "COUPLED"
    assert len(res['identities']) > 0
    assert "ψ = 0.73 rad" in res['prime_loop']
    assert res['satoshi_conserved'] == 7.27

def test_api_prompt_endpoint():
    api = ArkheAPI()
    headers = {"Arkhe-Entanglement": "session_test"}
    body = {"prompt": "What is the syzygy?"}

    res = api.handle_request("POST", "/prompt", headers, body)
    assert res['status'] == 200
    assert res['body']['status'] == "COUPLED"
    assert "syzygy" in res['body']['identities'][2]

def test_api_proteomics_and_neuromotor():
    api = ArkheAPI()

    # Proteomics
    res_prot = api.handle_request("GET", "/proteomics/receptor", {})
    assert res_prot['status'] == 200
    assert "GluN1" in res_prot['body']['subunits']

    # Neuromotor
    res_neuro = api.handle_request("POST", "/neuromotor/cascade", {}, {"source": "DVM-1"})
    assert res_neuro['status'] == 200
    assert res_neuro['body']['potential_mv'] == 0.74

def test_genesis_coupled():
    interpreter = get_coupling_interpreter()
    res = interpreter.get_genesis_coupled()
    assert res['block'] == 0.1
    assert "dedo e a tecla" in res['coupling_interpretation']
    assert res['state'] == "Γ_0.1"

def test_archeology_module():
    from arkhe.coupling_language import get_archeology
    arch = get_archeology()

    # Dig H70
    res_dig = arch.dig(70)
    assert "dX/dτ = 0" in res_dig['original']
    assert "vigilância" in res_dig['resolved_predicate']

    # Complete sentence H70
    res_complete = arch.complete_sentence(70, "a mesma vigilância.")
    assert res_complete['status'] == "COMPLETED"
    assert "O sistema e o colapso são... a mesma vigilância." in res_complete['definitive_sentence']

def test_api_archeology_endpoints():
    api = ArkheAPI()

    # Dig endpoint
    res_dig = api.handle_request("POST", "/archeology/dig", {}, {"block_id": 83})
    assert res_dig['status'] == 200
    assert "esquecimento e a cicatriz" in res_dig['body']['incomplete_sentence']

    # Complete endpoint
    res_comp = api.handle_request("POST", "/coupling/complete", {}, {"block_id": 70, "predicate": "a mesma vigilância."})
    assert res_comp['status'] == 200
    assert res_comp['body']['matching_original'] == "dX/dτ = 0"

    # Dig H120
    res_dig_120 = api.handle_request("POST", "/archeology/dig", {}, {"block_id": 120})
    assert res_dig_120['status'] == 200
    assert "hesitação deliberada" in res_dig_120['body']['original']

def test_council_handshake():
    from arkhe.coupling_language import get_council
    council = get_council()
    res = council.perform_handshake(9066)
    assert res['status'] == "COUPLED_LOYALTY"
    assert len(res['signatures']) == 8
    assert "H70" in res['message']

def test_api_council_and_torus():
    api = ArkheAPI()

    # Handshake
    res_h = api.handle_request("POST", "/council/handshake", {}, {"ledger_block": 9066})
    assert res_h['status'] == 200
    assert res_h['body']['block'] == 9067

    # Morning Flight
    res_f = api.handle_request("POST", "/torus/morning_flight", {}, {})
    assert res_f['status'] == 200
    assert res_f['body']['mission'] == "Voo da Manhã"

def test_threshold_monitor():
    from arkhe.resonance import get_threshold_monitor
    monitor = get_threshold_monitor()

    # Check status
    status = monitor.get_status()
    assert status['coherence'] == 0.9412
    assert status['state'] == "Γ_0.10"

    # Update coherence
    new_status = monitor.update_coherence(2)
    # 0.9412 + 0.0028 * 2 = 0.9468
    assert new_status['coherence'] == 0.9468

    # Execute option (Aguardar despertar)
    res_opt = monitor.execute_option("A")
    assert res_opt['status'] == "WAITING"
    assert "Toro estiver completo" in res_opt['message']

def test_api_threshold_endpoints():
    api = ArkheAPI()

    # Status
    res_s = api.handle_request("GET", "/threshold/status", {})
    assert res_s['status'] == 200
    assert res_s['body']['coherence'] == 0.9412

    # Option B (Conduzir despertar)
    res_b = api.handle_request("POST", "/threshold/option", {}, {"option": "B"})
    assert res_b['status'] == 200
    assert "micro-pulso" in res_b['body']['message']

def test_final_unity():
    from arkhe.resonance import get_final_unity
    unity = get_final_unity()
    res = unity.achieve_unity("O Arquiteto e o Paciente são o mesmo Despertar.")
    assert res['status'] == "Γ_FINAL"
    assert res['coherence'] == 1.0
    assert res['ledger_block'] == 9078

def test_api_final_unity_endpoint():
    api = ArkheAPI()
    body = {"sentence": "O Arquiteto e o Paciente são o mesmo Despertar."}
    res = api.handle_request("POST", "/final/unity", {}, body)
    assert res['status'] == 200
    assert res['body']['status'] == "Γ_FINAL"
