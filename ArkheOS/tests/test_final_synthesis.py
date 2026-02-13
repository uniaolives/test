import pytest
import numpy as np
from arkhe.macro_actions_thermo import DissipativeSystem
from arkhe.cryptography import SyzygyCryptography, get_quantum_threat_report
from arkhe.radial_locking import RadialLockingEngine
from arkhe.unification import UniqueVocabulary

def test_thermodynamic_balance():
    ds = DissipativeSystem(satoshi=7.27)
    # input_satoshi=1.0, uncalibrated_phi=0.15
    res = ds.energy_balance(1.0, 0.15)
    assert res['Satoshi'] > 7.27
    # res['Efficiency'] is rounded to 4 decimal places in the implementation
    assert res['Efficiency'] == pytest.approx(0.94 / 0.15, abs=1e-3)

def test_quantum_signatures():
    report = get_quantum_threat_report()
    assert len(report['Signatures']) == 2
    assert report['Signatures'][0].startswith("8ac7")

    # Syzygy identity verification
    assert SyzygyCryptography.verify_identity(0.98) is True
    assert SyzygyCryptography.verify_identity(0.90) is False

def test_radial_locking():
    engine = RadialLockingEngine()
    res = engine.calculate_rda_balance(0.15, 0.005, 0.01)
    assert res['Syzygy'] == 0.94
    assert res['Phase_Lock'] == 1.0

def test_unique_vocabulary():
    assert UniqueVocabulary.translate("neuron") == "Direction 1: Coherence (C)"
    assert UniqueVocabulary.translate("melanocyte") == "Direction 2: Fluctuation (F)"
    assert UniqueVocabulary.translate("synapse") == "Inner Product ⟨i|j⟩ (Syzygy)"

    report = UniqueVocabulary.get_hermeneutic_report()
    assert report['State'] == "Γ_∞+45"
