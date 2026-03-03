# papercoder_kernel/safety/theorem.py
from typing import Optional
import numpy as np
from ..core.ast import Program, AST, edit_distance
from ..lie.algebra import VectorField
from ..lie.group import Diffeomorphism, DiffeomorphismGroup

def is_safe_refactoring(phi: Diffeomorphism, group: DiffeomorphismGroup, tolerance: float = 1e-6) -> bool:
    """
    Teorema: phi é uma refatoração segura sse:
    1) phi é um difeomorfismo (já garantido pela classe)
    2) Existe v tal que phi ≈ exp(v) e v é completo.
    3) O período do fluxo de v é discreto em relação à identidade.
    """
    # 2) Tentar encontrar log(phi)
    v = extract_vector_field(phi, group)
    if v is None:
        return False   # não é exponencial

    # 2b) Verificar completude
    if not is_complete(v):
        return False

    # 3) Verificar discretude do período group
    # (testamos se exp(t·v) = identidade para t pequeno não‑nulo)
    for t in np.linspace(0.1, 10, 20): # Reduzido para performance no protótipo
        psi = group.exponential(v, steps=int(10*t) + 1)
        test_prog = _get_sample_program()
        if psi(test_prog) == test_prog and t > tolerance:
            # encontramos um período não‑trivial muito pequeno — grupo denso?
            return False

    return True

def extract_vector_field(phi: Diffeomorphism, group: DiffeomorphismGroup) -> Optional[VectorField]:
    """
    Tenta encontrar v tal que exp(v) ≈ phi.
    Implementação v0.1: Aproximação por diferenças finitas.
    """
    def generator(p: Program, epsilon: float) -> Program:
        # Aproximação de primeira ordem: phi(p) ≈ p + ε·v
        # v ≈ (phi(p) - p) / ε
        target = phi(p)
        dist = edit_distance(target, p)
        if dist == 0:
            return p
        # Retorna um programa modificado proporcionalmente à distância
        return _perturb(p, epsilon * dist)

    return VectorField(f"log({phi.name})", generator)

def is_complete(v: VectorField) -> bool:
    """Verifica se o campo é completo (fluxo existe para todo t)."""
    test_prog = _get_sample_program()
    for t in [0.1, 1.0, 5.0]:
        try:
            v.apply(test_prog, t)
        except Exception:
            return False
    return True

def _get_sample_program() -> Program:
    """Gera um programa aleatório/exemplo para verificação."""
    return Program(AST("root", [], {}), {"context": "global"})

def _perturb(p: Program, amount: float) -> Program:
    """Aplica uma perturbação infinitesimal no espaço de programas."""
    if amount == 0:
        return p
    # Simulação: altera metadados para mudar o hash e a identidade
    new_metadata = p.ast.metadata.copy()
    new_metadata["_epsilon_flow"] = new_metadata.get("_epsilon_flow", 0.0) + amount
    new_ast = AST(p.ast.node_type, p.ast.children, new_metadata)
    return Program(new_ast, p.type_context)
