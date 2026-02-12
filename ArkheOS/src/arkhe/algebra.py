"""
Arkhe(n) Vector Algebra Module
Implementation of the vec3 type (Γ_9041 - Γ_9043).
"""

from dataclasses import dataclass
import math
import cmath

@dataclass
class vec3:
    x: float
    y: float
    z: float
    C: float = 0.86
    F: float = 0.14
    omega: float = 0.00
    satoshi: float = 7.27

    def __post_init__(self):
        # Invariante fundamental: C + F = 1.0
        total = self.C + self.F
        if abs(total - 1.0) > 0.001:
            self.C /= total
            self.F /= total

    def norm(self) -> float:
        """Norma Arkhe(n): ‖v‖_A = √(x²·C + y²·C + z²·C) · (1 - F)"""
        # Simplificação: usando C como peso escalar para todas as coordenadas
        inner_sq = (self.x**2 + self.y**2 + self.z**2) * self.C
        return math.sqrt(inner_sq) * (1 - self.F)

    @staticmethod
    def add(a: 'vec3', b: 'vec3') -> 'vec3':
        """Adição vetorial (apenas se ω_a = ω_b)."""
        if abs(a.omega - b.omega) > 1e-6:
            raise ValueError(f"Inconsistent omega: {a.omega} != {b.omega}")

        r_x = a.x + b.x
        r_y = a.y + b.y
        r_z = a.z + b.z

        norm_a = a.norm()
        norm_b = b.norm()

        # Coerência resultante: média ponderada pelas normas
        if (norm_a + norm_b) > 0:
            r_C = (norm_a * a.C + norm_b * b.C) / (norm_a + norm_b)
        else:
            r_C = (a.C + b.C) / 2

        return vec3(
            x=r_x, y=r_y, z=r_z,
            C=r_C, F=1.0 - r_C,
            omega=a.omega,
            satoshi=a.satoshi + b.satoshi
        )

    @staticmethod
    def inner(a: 'vec3', b: 'vec3') -> complex:
        """Produto interno semântico (complexo se ω diferentes)."""
        # ⟨a|b⟩ = (a.x·b.x + a.y·b.y + a.z·b.z) · (1 - |ω_a - ω_b|/ω_max) · √(a.C·b.C) · exp(i·(φ_a - φ_b))
        omega_max = 0.10
        dot_product = (a.x * b.x + a.y * b.y + a.z * b.z)

        omega_diff = abs(a.omega - b.omega)
        omega_factor = max(0, 1 - omega_diff / omega_max)

        coherence_factor = math.sqrt(a.C * b.C)

        # fase Larmor simplificada: φ = 0.73 * (ω/0.07) se ω=0.07, ou apenas 0.73 se diff
        phase = 0.73 if omega_diff > 0.001 else 0.0

        magnitude = dot_product * omega_factor * coherence_factor
        return cmath.rect(magnitude, phase)

def vec3_gradient_coherence(x, y, z, radius=1.0):
    """Gradiente de coerência espacial (simulado)."""
    # |∇C|² ≈ 0.0049 no ponto (55.2, -8.3, -10.0)
    return vec3(x=0.07, y=0.0, z=0.0, C=0.0049, F=0.9951, omega=0.0)
