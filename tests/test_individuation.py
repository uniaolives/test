
import pytest
import numpy as np
from avalon.analysis.individuation import IndividuationManifold
from avalon.core.boot_filter import IndividuationBootFilter

def test_individuation_formula():
    manifold = IndividuationManifold()
    # I = F * (λ1/λ2) * (1 - S) * e^(i∮φdθ)
    F = 1.0
    l1, l2 = 0.5, 0.5
    S = 1.0
    phase = np.exp(1j * np.pi)

    I = manifold.calculate_individuation(F, l1, l2, S, phase)
    # Since S=1.0, (1-S) = 0.0, so I should be 0.0
    assert np.abs(I) == pytest.approx(0.0)

def test_individuation_classification():
    manifold = IndividuationManifold()
    I_optimal = 1.5 + 0j
    classification = manifold.classify_state(I_optimal)
    assert classification['state'] == 'OPTIMAL_INDIVIDUATION'
    assert classification['risk'] == 'LOW'

@pytest.mark.asyncio
async def test_boot_filter():
    arkhe = {'C': 0.9, 'I': 0.9, 'E': 0.9, 'F': 0.1} # Low purpose
    filter_obj = IndividuationBootFilter(arkhe)

    res = await filter_obj.apply_filter("Test Phase")
    assert res['status'] == "PROTECTED"
    # Should have auto-corrected F
    assert arkhe['F'] > 0.1
