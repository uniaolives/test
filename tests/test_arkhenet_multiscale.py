import sys
import os
import numpy as np
import pytest

# Add ArkheOS src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.interfaces.quantum_bio import QuantumBioInterface
from arkhe.interfaces.bio_tech import BioTechInterface
from arkhe.interfaces.quantum_tech import QTechInterface
from arkhe.interfaces.tri_hybrid import TriHybridNode

def test_quantum_bio_interface():
    qbio = QuantumBioInterface()
    result = qbio.simulate_handover(excitation_power=1.0, steps=10)
    assert result['total_released'] > 0
    assert result['final_load'] < 100.0

def test_bio_tech_interface():
    biotech = BioTechInterface()
    result = biotech.execute_mission()
    assert result['success'] is True
    assert result['particles_injected'] == 10000
    assert result['final_coverage'] > 0.5

def test_quantum_tech_interface():
    qtech = QTechInterface(n_drones=2)
    results = qtech.establish_quantum_network()
    assert len(results) == 1
    assert results[0]['success'] is True
    assert results[0]['key_length'] > 0

def test_tri_hybrid_node():
    node = TriHybridNode("TEST-NODE")
    result = node.run_mission(n_cycles=10)
    assert result['final_coherence'] > 0
    assert result['total_drug_delivered'] > 0

if __name__ == "__main__":
    # Manual execution if needed
    pytest.main([__file__])
