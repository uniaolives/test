# papercoder_kernel/types/dependent.py
from __future__ import annotations
from typing import Generic, TypeVar, Callable
from dataclasses import dataclass
from ..core.ast import Program

P = TypeVar('P', bound=Program)

@dataclass
class Refactor(Generic[P]):
    """
    Tipo dependente: Refactor a b representa uma refatoração de a para b.
    Em Python, simulamos isso garantindo as instâncias src e dst no dataclass.
    """
    src: P
    dst: P
    mapping: Callable[[P], P]   # deve satisfazer mapping(src) == dst
    proof: Callable[[], bool]    # prova de que a semântica é preservada

    def compose(self, other: Refactor[P]) -> Refactor[P]:
        """Composição sequencial de refatorações."""
        if self.dst != other.src:
            raise TypeError(f"Incompatible types for composition: dst {self.dst} != src {other.src}")

        return Refactor(
            self.src,
            other.dst,
            lambda x: other.mapping(self.mapping(x)),
            lambda: self.proof() and other.proof()
        )

    @staticmethod
    def identity(p: P) -> Refactor[P]:
        """Refatoração identidade (elemento neutro)."""
        return Refactor(p, p, lambda x: x, lambda: True)
