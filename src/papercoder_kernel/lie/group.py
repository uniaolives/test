# papercoder_kernel/lie/group.py
from typing import Callable, Optional
from ..core.ast import Program
from .algebra import VectorField

class Diffeomorphism:
    """Elemento do grupo de difeomorfismos (refatoração finita)."""
    def __init__(self, name: str, mapping: Callable[[Program], Program]):
        self.name = name
        self.mapping = mapping
        self.inverse: Optional['Diffeomorphism'] = None

    def __call__(self, p: Program) -> Program:
        return self.mapping(p)

    def compose(self, other: 'Diffeomorphism') -> 'Diffeomorphism':
        """Composição de difeomorfismos (multiplicação do grupo)."""
        # Note: Usually phi1 o phi2 means phi1(phi2(p))
        return Diffeomorphism(
            f"{self.name}∘{other.name}",
            lambda p: self.mapping(other.mapping(p))
        )

    def set_inverse(self, inv: 'Diffeomorphism'):
        self.inverse = inv
        if inv.inverse is None:
            inv.inverse = self

class DiffeomorphismGroup:
    """Grupo de Lie de difeomorfismos do espaço de programas."""
    def __init__(self):
        self.identity = Diffeomorphism("id", lambda p: p)
        self.identity.set_inverse(self.identity)

    def exponential(self, v: VectorField, steps: int = 100) -> Diffeomorphism:
        """
        Mapa exponencial: exp: 𝔤 → G
        Integra um campo vetorial em um difeomorfismo finito.
        """
        def flow(p: Program) -> Program:
            current = p
            dt = 1.0 / steps
            for _ in range(steps):
                current = v.apply(current, dt)
            return current
        return Diffeomorphism(f"exp({v.name})", flow)
