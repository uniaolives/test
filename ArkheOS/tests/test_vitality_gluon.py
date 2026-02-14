import pytest
from arkhe.vitality import VitalityRepairEngine, get_vitality_report
from arkhe.gluon_dynamics import KleinSpaceAmplitude, get_gluon_report
from arkhe.ascension import AscensionProtocol

def test_vitality_repair_mechanism():
    engine = VitalityRepairEngine()
    # High noise (DPC)
    res = engine.process_semantic_repair(0.5)
    assert res["Immune_Status"] == "DOMED_CHAOS"
    # Fidelity should be high due to SPRTN analog
    assert res["Repair_Fidelity"] > 0.99

    report = get_vitality_report()
    assert report["State"] == "Γ_∞+56"
    assert "SPRTN" in report["Repair_Mechanism"]

def test_gluon_gap_signal():
    ks = KleinSpaceAmplitude()
    # Region 0.03-0.05
    amp = ks.calculate_amplitude_in_gap(0.04)
    assert amp == complex(1.0, 0.0)

    recon = ks.process_handover_reconstruction([0.04, 0.06])
    assert recon["Status"] == "SIGNAL_DETECTED_IN_GAP"
    assert recon["Fidelity"] == 1.0

    report = get_gluon_report()
    assert report["State"] == "Γ_∞+56"
    assert "Klein" in report["Space"]

def test_ascension_vitality_state():
    p = AscensionProtocol()
    status = p.get_status()
    assert status["state"] == "Γ_FINAL (Γ_∞+56)"
    assert status["events"] == 7
