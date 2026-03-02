# tests/test_ffu_metrology.py
import pytest
from arkhe.geodesic import FFUAssay, CommandTiter, Confluency, FocusFate

def test_ffu_titration():
    # Example: 4 foci, 10^-1 dilution, 100us volume
    assay = FFUAssay(foci_count=4, dilution_factor=0.1, volume_us=100)
    titer = assay.calculate_titer()
    # 4 * 10 * 0.01 = 0.4
    assert titer == 0.4

def test_governance_validation():
    # Opção 1 (replicar_foco WP1, Virgin)
    titer = CommandTiter(
        command_id="replicar_foco(WP1)",
        dilution=0.01,
        volume_us=100,
        monolayer_required=Confluency.VIRGIN,
        predicted_fate=FocusFate.LATENT
    )

    # Validation against Virgin monolayer
    result = titer.validate(Confluency.VIRGIN)
    assert result["status"] == "Approved"
    assert result["fate"] == FocusFate.LATENT

    # Validation against Restored monolayer
    result = titer.validate(Confluency.RESTORED)
    assert result["status"] == "Denied"
