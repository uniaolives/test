# papercoder_kernel/lie/algebra.py
from typing import Callable, Optional
from ..core.ast import Program

class VectorField:
    """
    Campo vetorial no espaço de programas.
    Representa uma direção de refatoração infinitesimal.
    """
    def __init__(self, name: str, generator: Callable[[Program, float], Program]):
        self.name = name
        self.generator = generator   # v(p, ε) ≈ exp(ε·v)(p)
        self.completeness: Optional[bool] = None   # se é completo (i.e., integrável para todo t)

    def apply(self, p: Program, epsilon: float) -> Program:
        """Aplica o fluxo infinitesimal com passo epsilon."""
        # Aplicação infinitesimal: aproximação de exp(ε·v)(p)
        return self.generator(p, epsilon)
