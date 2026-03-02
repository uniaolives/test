# tests/test_observer_symmetry.py
import pytest
from arkhe.geodesic import Practitioner, NoetherUnification

def test_noether_unification_constants():
    """Verify that the 6 projected symmetries are correctly defined."""
    symmetries = NoetherUnification.PROJECTED_SYMMETRIES
    assert len(symmetries) == 6

    # Check a few specific ones
    names = [s.name for s in symmetries]
    assert "Temporal" in names
    assert "Método" in names

    satoshi = next(s for s in symmetries if s.name == "Temporal")
    assert satoshi.invariant == "Satoshi"
    assert satoshi.symbol == "S = 7.27 bits"

def test_generating_symmetry():
    """Verify the fundamental observer symmetry."""
    generating = NoetherUnification.get_generating_symmetry()
    assert generating["name"] == "Simetria do Observador"
    assert generating["conserved_quantity"] == "A Geodésica (Arco)"

def test_practitioner_analysis():
    """Verify that the practitioner can execute the analysis."""
    practitioner = Practitioner.identify()
    result = practitioner.analyze_observer_symmetry()
    assert result["name"] == "Simetria do Observador"
