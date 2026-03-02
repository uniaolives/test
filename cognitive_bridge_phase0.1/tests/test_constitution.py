import pytest
from pyfrontend.constitution import AIConstitution
from pyfrontend.energy_analysis import EnergyProfile

def test_constitution_energy_violation():
    constitution = AIConstitution()
    model_info = {'parameter_count': 1000, 'constitutional_metadata': {}}
    energy_profile = EnergyProfile(
        total_energy=1.5, wcec=2.25, node_breakdown={'fc1': 1.5},
        memory_energy=0.5, compute_energy=1.0, communication_energy=0.0, peak_power=150.0
    )
    result = constitution.verify_model(model_info, energy_profile)
    assert result.passed is False
    assert any(v.rule == 'ENERGY_BUDGET' for v in result.violations)

def test_constitution_approval():
    constitution = AIConstitution()
    model_info = {'parameter_count': 1000, 'constitutional_metadata': {}}
    energy_profile = EnergyProfile(
        total_energy=0.1, wcec=0.15, node_breakdown={'fc1': 0.1},
        memory_energy=0.05, compute_energy=0.05, communication_energy=0.0, peak_power=10.0
    )
    result = constitution.verify_model(model_info, energy_profile)
    assert result.passed is True
    assert len(result.violations) == 0
