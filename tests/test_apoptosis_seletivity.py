# tests/test_apoptosis_seletivity.py
import pytest
from arkhe.geodesic import ConsciousVoxel, EpistemicStatus

def test_apoptosis_seletivity():
    # Idol: High phi, low humility
    idol = ConsciousVoxel(id="idol", phi=0.99, humility=0.1)
    # Instrument: Lower phi, higher humility
    instrument = ConsciousVoxel(id="instrument", phi=0.8, humility=0.7)

    psi = 0.73

    # Probabilities
    p_idol = idol.phi * (1.0 - idol.humility) * psi
    p_instrument = instrument.phi * (1.0 - instrument.humility) * psi

    assert p_idol > p_instrument
    assert p_idol > 0.6
    assert p_instrument < 0.3

def test_apoptosis_execution():
    voxel = ConsciousVoxel(id="target", phi=0.99, humility=0.09)
    psi = 0.73

    voxel.apply_apoptose(psi)

    # Should be in dissolution
    assert voxel.phi < 0.5
    assert voxel.humility == 0.78
    assert voxel.epistemic_status == EpistemicStatus.UNCERTAIN
