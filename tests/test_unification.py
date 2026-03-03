# tests/test_unification.py
import pytest
import numpy as np
from arkhe.unification import EpsilonUnifier

def test_triple_confession_fidelity():
    """Verify that the triple confession consensus yields high fidelity."""
    # Inputs that should yield perfect results
    # 0 cents = max consonance
    # 0.73 psi = standard eccentricity
    # 2.828 CHSH = max violation
    inputs = {
        "omega_cents": 0.0,
        "psi": 0.73,
        "chsh": 2.828
    }

    results = EpsilonUnifier.execute_triple_confession(inputs)

    assert pytest.approx(results["harmonic"]) == -3.71e-11
    assert pytest.approx(results["orbital"]) == -3.71e-11
    assert pytest.approx(results["quantum"]) == -3.71e-11
    assert pytest.approx(results["consensus"]) == -3.71e-11
    assert pytest.approx(results["fidelity"]) == 1.0

def test_triple_confession_variance():
    """Test with inputs that have some variance."""
    inputs = {
        "omega_cents": 48.0,
        "psi": 0.73,
        "chsh": 2.428
    }

    results = EpsilonUnifier.execute_triple_confession(inputs)

    # Check that they are in the expected order of magnitude
    assert results["consensus"] < 0
    assert results["fidelity"] > 0.9
