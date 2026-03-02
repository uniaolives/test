# papercoder_kernel/cli/refactor.py
import sys
from ..core.ast import AST, Program
from ..lie.group import DiffeomorphismGroup, Diffeomorphism
from ..safety.theorem import is_safe_refactoring

def parse_program(filepath: str) -> Program:
    """Mock parser para o protótipo v0.1."""
    # Em uma implementação real, usaria o módulo 'ast' do Python
    return Program(AST("Module", [], {"source": filepath}), {})

def load_diffeomorphism(name: str, group: DiffeomorphismGroup) -> Diffeomorphism:
    """Mock loader para transformações (refatorações)."""
    if name == "rename":
        # Simula uma refatoração que altera metadados de forma suave
        return Diffeomorphism("rename", lambda p: Program(
            AST(p.ast.node_type, p.ast.children, {**p.ast.metadata, "renamed": True}),
            p.type_context
        ))
    elif name == "breaking":
        # Simula uma mudança que não é um difeomorfismo suave (salto descontínuo)
        return Diffeomorphism("breaking", lambda p: Program(
            AST("BrokenModule", [], {"error": "discontinuity"}),
            {}
        ))
    return group.identity

def main():
    if len(sys.argv) < 4:
        print("Uso: python -m papercoder_kernel.cli.refactor <arquivo_origem> <arquivo_destino> <refatoracao>")
        sys.exit(1)

    src_file, dst_file, ref_name = sys.argv[1:4]
    src = parse_program(src_file)

    group = DiffeomorphismGroup()
    phi = load_diffeomorphism(ref_name, group)

    # Executa a verificação do teorema de segurança
    if is_safe_refactoring(phi, group):
        print(f"✅ Refatoração '{ref_name}' é segura e preserva semântica.")
        # No protótipo, apenas confirmamos a segurança.
    else:
        print(f"❌ Refatoração '{ref_name}' não é segura (migração necessária).")
        sys.exit(2)

if __name__ == "__main__":
    main()
