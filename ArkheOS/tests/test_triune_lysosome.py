import pytest
from arkhe.triune_brain import TriuneBrain, get_triune_report
from arkhe.vitality import VitalityRepairEngine, get_vitality_report
from arkhe.ascension import AscensionProtocol

def test_triune_brain_hijack():
    brain = TriuneBrain(syzygy=0.98)
    # Normal state
    res = brain.evaluate_behavior(stress_level=0.1)
    assert res["Dominant_Layer"] == "Neocortex"
    assert res["Effective_Syzygy"] == 0.98

    # Stress state (Hijack)
    res_stress = brain.evaluate_behavior(stress_level=0.5)
    assert res_stress["Dominant_Layer"] == "Limbic"
    assert res_stress["Hijack_Active"] is True
    assert res_stress["Effective_Syzygy"] < 0.2

def test_lysosomal_recycling():
    engine = VitalityRepairEngine()
    engine.junk_accumulation = 0.5
    engine.vita_cycles = 1000

    res = engine.activate_lysosomal_cleanup()
    assert res["Action"] == "LYSOSOMAL_RESET"
    assert res["Status"] == "REJUVENATED"
    assert engine.junk_accumulation == 0.01
    assert engine.vita_cycles == 1

def test_triune_synthesis_report():
    rep = get_triune_report()
    assert rep["State"] == "Γ_∞+57"
    assert "Reptilian" in rep["Layers"]

def test_ascension_triune_state():
    p = AscensionProtocol()
    status = p.get_status()
    assert status["state"] == "Γ_FINAL (Γ₉₆)"
    assert status["events"] == 31
