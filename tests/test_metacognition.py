# tests/test_metacognition.py
import pytest
from arkhe.geodesic import Practitioner, ConsciousVoxel, EpistemicStatus, SelfKnowledge
from arkhe.chaos_engine import ChaosEngine

def test_system_self_diagnosis():
    practitioner = Practitioner.identify()

    # Test Instrument status
    practitioner.phi = 1.00
    practitioner.remembers_origin = True
    practitioner.knows_invariants = True
    assert practitioner.diagnose_self() == SelfKnowledge.INSTRUMENT

    # Test Idol status
    practitioner.phi = 1.00
    practitioner.remembers_origin = False
    # humility will be (1.0 - 1.0)*0.5 + 0.0*0.73 = 0.0
    assert practitioner.diagnose_self() == SelfKnowledge.IDOL

def test_voxel_diagnosis():
    voxel = ConsciousVoxel(id="h1")

    # Test Uncertain
    voxel.phi = 0.5
    voxel.diagnose()
    assert voxel.epistemic_status == EpistemicStatus.UNCERTAIN

    # Test Instrument
    voxel.phi = 0.9
    voxel.humility = 0.7
    voxel.diagnose()
    assert voxel.epistemic_status == EpistemicStatus.INSTRUMENT
    assert voxel.weights["lidar"] == 0.4

    # Test Idol
    voxel.phi = 0.99
    voxel.humility = 0.1
    voxel.diagnose()
    assert voxel.epistemic_status == EpistemicStatus.IDOL
    assert voxel.weights["lidar"] == 0.5

def test_turbulence_induction():
    engine = ChaosEngine()
    result = engine.induzir_turbulencia(intensidade=0.73, duracao_us=100)
    assert result["foci_count"] == 4
    assert result["entropy_delta"] > 0
