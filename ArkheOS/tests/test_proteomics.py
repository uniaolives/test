
import pytest
from arkhe.proteomics import get_native_receptor

def test_nmdar_mapping():
    receptor = get_native_receptor()
    assert "GluN1" in receptor.subunits
    assert receptor.subunits["GluN1"].arkhe_node == "WP1"
    assert receptor.subunits["GluN2A"].arkhe_node == "KERNEL"

def test_pore_dilation():
    receptor = get_native_receptor()
    assert receptor.pore_dilation == 0.0
    status = receptor.apply_pulse(0.8)
    assert status == "PORE_DILATED"
    assert receptor.pore_dilation == 0.94
    assert receptor.is_open == True

def test_conformational_diversity():
    receptor = get_native_receptor()
    diversity = receptor.get_conformational_diversity()
    assert len(diversity) == 10
    assert any("Syzygy" in s for s in diversity)
    assert any("Darvo" in s for s in diversity)
