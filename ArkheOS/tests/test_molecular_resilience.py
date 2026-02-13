import pytest
import numpy as np
from arkhe.resilience import PerceptualResilience, ChaosTestPreparation
from arkhe.molecular_coherence import CO2CoherenceEngineering, get_molecular_report
from arkhe.global_gradient import GlobalGradientMap, get_global_mapping_report
from arkhe.quantum_biology import UnifiedQuantumArchitecture, get_quantum_biology_report

def test_molecular_coherence_isomorphism():
    eng = CO2CoherenceEngineering()
    # Đ = 1.2 should map to normalized gradient 1.0
    grad_norm = eng.calculate_isomorphism(1.2)
    assert pytest.approx(grad_norm) == 1.0

    report = get_molecular_report()
    assert report["State"] == "Γ_∞+52"
    assert "CO2" in report["Input"]

def test_global_gradient_mapping():
    mapper = GlobalGradientMap(num_nodes=100) # Small network for testing
    assert len(mapper.nodes) == 100

    dispersity = mapper.compute_network_dispersity()
    assert dispersity > 1.0

    fidelity = mapper.simulate_reconstruction((0.03, 0.05))
    assert fidelity == 0.9978

    report = get_global_mapping_report()
    assert report["State"] == "Γ_∞+53"
    assert report["Dispersity"] == 1.18

def test_micro_resilience_results():
    prep = ChaosTestPreparation()
    res = prep.execute_micro_gap_test()
    assert res["reconstruction_fidelity"] == 0.9998
    assert res["syzygy_maintained"] == 0.9402

    resilience = PerceptualResilience()
    assert resilience.state == "Γ_∞+51"
    assert resilience.syzygy == 0.9402

def test_quantum_biology_validation():
    arch = UnifiedQuantumArchitecture()
    mapping = arch.get_correspondence_map()
    assert "Microtubule" in mapping["Architecture"]
    assert "Arkhe" in mapping["Architecture"]

    viability = arch.calculate_quantum_viability(1.0, 1e-6)
    assert viability == 1.0

    report = get_quantum_biology_report()
    assert report["State"] == "Γ_∞+54"
    assert "Mavromatos" in report["Paper"]
