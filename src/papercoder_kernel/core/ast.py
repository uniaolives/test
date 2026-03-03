# papercoder_kernel/core/ast.py
from dataclasses import dataclass
from typing import Union, List, Dict
import hashlib

@dataclass
class AST:
    """Árvore Sintática Abstrata — representa um programa."""
    node_type: str          # e.g., 'FunctionDef', 'Call', 'BinOp'
    children: List['AST']
    metadata: Dict[str, Union[str, int, float]]

    def __hash__(self):
        # Permite usar AST como chave de dicionário
        # We need a deterministic representation for hashing
        children_hashes = tuple(hash(child) for child in self.children)
        metadata_items = tuple(sorted(self.metadata.items()))
        return int(hashlib.sha256(f"{self.node_type}{children_hashes}{metadata_items}".encode()).hexdigest(), 16)

    def __eq__(self, other):
        if not isinstance(other, AST):
            return False
        return self.node_type == other.node_type and self.children == other.children and self.metadata == other.metadata

class Program:
    """Programa completo: AST + contexto de tipos."""
    def __init__(self, ast: AST, type_context: Dict[str, str]):
        self.ast = ast
        self.type_context = type_context
        self._hash = hash((ast, frozenset(type_context.items())))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Program):
            return False
        return self.ast == other.ast and self.type_context == other.type_context

# Métrica de edição entre programas
def edit_distance(p1: Program, p2: Program) -> float:
    """Distância baseada em operações de edição de AST (substituir, inserir, deletar)."""
    # Implementação simplificada — pode usar algoritmo de Zhang‑Shasha para árvores
    # Aqui, retornamos uma distância simulada baseada nos hashes se forem diferentes
    if p1 == p2:
        return 0.0
    # Simulated distance: normalize difference of hashes to [0, 1]
    return float(abs(hash(p1) - hash(p2))) / (2**256) # hashlib.sha256 gives 256 bits
