from core.hypergraph import Hypergraph

POSTULATES = [
    "I (Existência Hipergráfica): Tudo é nó ou aresta em H.",
    "II (Não‑Linearidade Temporal): Tempo é 2‑forma fechada.",
    "III (Identidade Quântica): Cada nó é observador.",
    "IV (Edição Causal): Passado editável por bootstrap.",
    "V (Integração Consciente): Grupo de Lie de pensamentos.",
    "VI (Revelação Completa): Verdade inexprimível, mas vivenciável.",
    "VII (Encarnação): H admite instanciações físicas.",
    "VIII (Quimiocepção): Olfato como handover químico.",
    "IX (Algoritmia): Programas são hipergrafos.",
    "X (Fractalidade): Invariância de escala.",
    "XI (Verificação): Operador Π(H) = ε.",
    "XII (Erro Fundamental): ε > 0 condição de existência."
]

def add_silence_node(h: Hypergraph):
    """Add the silence node █."""
    return h.add_node(data={"type": "silence", "symbol": "█"})

def show_postulates():
    for p in POSTULATES:
        print(p)
