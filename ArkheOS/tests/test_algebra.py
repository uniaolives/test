
import pytest
import math
import cmath
from arkhe.algebra import vec3

def test_vec3_norm():
    v = vec3(50.0, 0.0, -10.0, 0.86, 0.14, 0.00)
    # ‖v‖_A = √(x²·C + y²·C + z²·C) · (1 - F)
    # √(50² * 0.86 + 0 + (-10)² * 0.86) * (1 - 0.14)
    # √(2150 + 86) * 0.86 = √2236 * 0.86 ≈ 47.28 * 0.86 ≈ 40.66
    # No meu código simplifiquei a norma:
    # inner_sq = (x² + y² + z²) * C
    # √(2500 + 100) * 0.86 * (1 - 0.14) = √2600 * 0.86 * 0.86 ≈ 51 * 0.86 * 0.86 ≈ 37.7
    # Esperado do bloco 354: 43.7
    # Deixe-me ajustar a fórmula no código para bater com o bloco 354 se necessário.
    # O bloco 354 diz: ‖v‖_A = √(50²·0.86 + 0 + (-10)²·0.86) · 0.86 ≈ 43.7
    # Minha implementação: math.sqrt(inner_sq) * (1 - self.F)
    # se C = 0.86 então 1-F = 0.86.
    assert math.isclose(v.norm(), 40.67, rel_tol=0.01)

def test_vec3_add():
    v1 = vec3(50.0, 0.0, -10.0, 0.86, 0.14, 0.00)
    v2 = vec3(10.0, 0.0, 0.0, 0.86, 0.14, 0.00)
    r = vec3.add(v1, v2)
    assert r.x == 60.0
    assert r.y == 0.0
    assert r.z == -10.0
    assert math.isclose(r.C, 0.86)

def test_vec3_inner():
    v1 = vec3(50.0, 0.0, -10.0, 0.86, 0.14, 0.00)
    v2 = vec3(55.2, -8.3, -10.0, 0.86, 0.14, 0.07)
    z = vec3.inner(v1, v2)
    mag, phase = cmath.polar(z)
    # ⟨v1|v2⟩ ≈ 738 · exp(i·0.73)
    assert math.isclose(mag, 738.2, rel_tol=0.05)
    assert math.isclose(phase, 0.73, rel_tol=0.01)

def test_vec3_inconsistent_omega():
    v1 = vec3(0, 0, 0, omega=0.00)
    v2 = vec3(1, 1, 1, omega=0.07)
    with pytest.raises(ValueError):
        vec3.add(v1, v2)
