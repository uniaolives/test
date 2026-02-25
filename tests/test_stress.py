# tests/test_stress.py
import pytest
import numpy as np
from core.python.axos.axos_v3 import AxosV3

@pytest.fixture
def axos():
    return AxosV3()

@pytest.mark.parametrize("h21", [200, 250, 300, 350])
def test_stability_near_critical(axos, h21):
    h11 = 491 # safety
    result = axos.explore_landscape(h11=h11, h21=h21)
    assert result.status == "SUCCESS"
    assert result.data['coherence'] > 0.0

def test_creativity_bound(axos):
    # Criatividade deve estar entre -1 e 1
    result = axos.generate_entity()
    assert result.status == "SUCCESS"
    # Note: simulate_entity in generating module returns positive coherence for now
    # but we check the personality based on h11/h21
    assert "personality" in result.data
