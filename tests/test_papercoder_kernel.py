import pytest
from papercoder_kernel.core.ast import AST, Program, edit_distance
from papercoder_kernel.lie.algebra import VectorField
from papercoder_kernel.lie.group import Diffeomorphism, DiffeomorphismGroup
from papercoder_kernel.types.dependent import Refactor
from papercoder_kernel.safety.theorem import is_safe_refactoring

def test_ast_equality():
    a1 = AST("func", [], {"name": "test"})
    a2 = AST("func", [], {"name": "test"})
    assert a1 == a2
    assert hash(a1) == hash(a2)

def test_program_distance():
    p1 = Program(AST("func", [], {}), {"x": "int"})
    p2 = Program(AST("func", [], {}), {"x": "int"})
    assert edit_distance(p1, p2) == 0.0

    p3 = Program(AST("func", [AST("return", [], {})], {}), {})
    assert edit_distance(p1, p3) > 0

def test_lie_group_composition():
    phi1 = Diffeomorphism("phi1", lambda p: Program(AST(p.ast.node_type, p.ast.children, {**p.ast.metadata, "a": 1}), p.type_context))
    phi2 = Diffeomorphism("phi2", lambda p: Program(AST(p.ast.node_type, p.ast.children, {**p.ast.metadata, "b": 2}), p.type_context))
    phi12 = phi1.compose(phi2)

    p = Program(AST("root", [], {}), {})
    res = phi12(p)
    assert res.ast.metadata["a"] == 1
    assert res.ast.metadata["b"] == 2

def test_dependent_types():
    pa = Program(AST("a", [], {}), {})
    pb = Program(AST("b", [], {}), {})
    pc = Program(AST("c", [], {}), {})

    r1 = Refactor(pa, pb, lambda p: pb, lambda: True)
    r2 = Refactor(pb, pc, lambda p: pc, lambda: True)

    r12 = r1.compose(r2)
    assert r12.src == pa
    assert r12.dst == pc

    with pytest.raises(TypeError):
        r1.compose(r1) # pb != pa

def test_safety_theorem():
    group = DiffeomorphismGroup()
    # A safe refactoring (smooth)
    phi_safe = Diffeomorphism("safe", lambda p: Program(
        AST(p.ast.node_type, p.ast.children, {**p.ast.metadata, "step": 0.1}),
        p.type_context
    ))
    # No protótipo, is_safe_refactoring deve retornar True para esta mudança suave
    assert is_safe_refactoring(phi_safe, group) == True

def test_exponential_map():
    group = DiffeomorphismGroup()
    v = VectorField("test_v", lambda p, eps: Program(
        AST(p.ast.node_type, p.ast.children, {**p.ast.metadata, "flow": p.ast.metadata.get("flow", 0) + eps}),
        p.type_context
    ))
    phi = group.exponential(v, steps=10)
    p = Program(AST("root", [], {"flow": 0}), {})
    res = phi(p)
    # 10 steps of 0.1 should sum to 1.0
    assert pytest.approx(res.ast.metadata["flow"]) == 1.0
