# tests/test_oncology_therapy.py
import pytest
from arkhe.geodesic import ConsciousVoxel, CannabinoidTherapy, Ligand, Receptor, EpistemicStatus

def test_cannabinoid_efficacy_on_idol():
    # Idol voxel: High phi, low humility
    voxel = ConsciousVoxel(id="idol_tumor", phi=0.99, humility=0.1)
    voxel.diagnose()
    assert voxel.epistemic_status == EpistemicStatus.IDOL

    therapy = CannabinoidTherapy(ligand=Ligand.THC, dose_ffu=10.0, receptors=[Receptor.CB1])
    efficacy = therapy.calculate_efficacy(voxel)

    # Efficacy should be high for idols: 0.9 * (1.0 - 0.1) = 0.81
    assert pytest.approx(efficacy) == 0.81

    original_phi = voxel.phi
    voxel.apply_therapy(therapy)

    # New phi should be original_phi * (1.0 - 0.81) = 0.99 * 0.19 = 0.1881
    assert pytest.approx(voxel.phi) == 0.1881
    assert voxel.epistemic_status == EpistemicStatus.UNCERTAIN

def test_cannabinoid_efficacy_on_instrument():
    # Instrument voxel: Moderate phi, high humility
    voxel = ConsciousVoxel(id="instrument_cell", phi=0.85, humility=0.7)
    voxel.diagnose()
    assert voxel.epistemic_status == EpistemicStatus.INSTRUMENT

    therapy = CannabinoidTherapy(ligand=Ligand.CBD, dose_ffu=30.0, receptors=[Receptor.CB2])
    efficacy = therapy.calculate_efficacy(voxel)

    # Efficacy should be low for instruments: 0.2 * 0.7 = 0.14
    assert pytest.approx(efficacy) == 0.14

    voxel.apply_therapy(therapy)
    # New phi: 0.85 * 0.86 = 0.731
    assert pytest.approx(voxel.phi) == 0.731
