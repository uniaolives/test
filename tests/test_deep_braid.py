# tests/test_deep_braid.py
import pytest
from papercoder_kernel.math.braid import DeepBraidArchitecture

def test_braid_word_generation():
    # p=3, M_3=7, Perfect=2^2 * 7 = 28
    # 28 in binary is 11100
    # dim=3, fios=3, generators σ_1, σ_2
    braid = DeepBraidArchitecture(3)
    word = braid.generate_braid_word()

    # bits of 28: '1', '1', '1', '0', '0'
    # indices: 0, 1, 2, 3, 4
    # i=0: '1' -> (0%2)+1 = 1 -> σ_1
    # i=1: '1' -> (1%2)+1 = 2 -> σ_2
    # i=2: '1' -> (2%2)+1 = 1 -> σ_1
    assert word == ["σ_1", "σ_2", "σ_1"]

def test_deep_braid_p61_stability():
    braid = DeepBraidArchitecture(61)
    assert braid.stability_check() == True

    invariants = braid.compute_invariants()
    assert invariants['stability'] > 0.99 # original ratio was M_p/2^p
    # Note: my stability_check uses 2^(p-1)/M_p, but invariants uses M_p/2^p as in snippet.

def test_invariants_format():
    braid = DeepBraidArchitecture(5)
    invariants = braid.compute_invariants()
    assert "q^5" in invariants['jones']
    assert "α^31" in invariants['homfly']
