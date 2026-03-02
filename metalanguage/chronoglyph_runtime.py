# chronoglyph_runtime.py
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional, Callable, Union
from collections import defaultdict
import networkx as nx
from enum import Enum, auto

class QuantumState(Enum):
    """Estados quânticos de um nó Chronoglyph"""
    SUPERPOSITION = auto()  # ◐ - múltiplos valores possíveis
    ENTANGLED = auto()      # ◎ - correlacionado com outros nós
    COLLAPSED = auto()      # ◉ - valor definido
    VOID = auto()           # ● - ausência, identidade

@dataclass
class ChronoNode:
    """Nó em um grafo Chronoglyph"""
    node_id: str
    kind: str  # 'core', 'orbit', 'state', 'possible', 'reference', 'void'
    value: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    entangled_with: Set[str] = field(default_factory=set)
    probability_amplitude: complex = 1.0 + 0j

class Universe:
    """Um universo paralelo na execução Chronoglyph"""
    def __init__(self, id: str, parent: Optional['Universe'] = None):
        self.id = id
        self.parent = parent
        self.state: Dict[str, Any] = {}  # Estado colapsado dos nós
        self.probability: float = 1.0
        self.history: List[str] = []  # Caminho de colapsos

    def collapse_node(self, node_id: str, value: Any):
        """Colapsa um nó neste universo"""
        self.state[node_id] = value
        self.history.append(f"{node_id}={value}")

    def branch(self, node_id: str, possibilities: List[Any]) -> List['Universe']:
        """Ramifica universo em múltiplos (superposição resolvida)"""
        branches = []
        n = len(possibilities)
        for i, val in enumerate(possibilities):
            new_uni = Universe(f"{self.id}.{i}", parent=self)
            new_uni.state = self.state.copy()
            new_uni.collapse_node(node_id, val)
            new_uni.probability = self.probability / n
            branches.append(new_uni)
        return branches

class ChronoGraph:
    """Grafo executável Chronoglyph"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, ChronoNode] = {}
        self.operadores = {
            'add': lambda a, b: a + b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None,
            'mul': lambda a, b: a * b,
            'fuse': lambda a, b: (a, b),  # ⊕ - tupla como superposição
            'merge': self._operator_merge,
        }

    def add_node(self, node: ChronoNode):
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            kind=node.kind,
            value=node.value,
            quantum=node.quantum_state
        )

    def add_connection(self, from_id: str, to_id: str, relation: str = 'associates'):
        if from_id in self.nodes and to_id in self.nodes:
            self.graph.add_edge(from_id, to_id, relation=relation)
            # Atualiza entanglement se apropriado
            if self.nodes[from_id].kind == 'reference':
                self.nodes[to_id].entangled_with.add(from_id)
                self.nodes[to_id].quantum_state = QuantumState.ENTANGLED

    def _operator_merge(self, a, b):
        """Operador de fusão heptapod: combina em superposição não-commutativa"""
        if isinstance(a, tuple) and isinstance(b, tuple):
            return a + b  # Concatenação de superposições
        return (a, b)  # Nova superposição

    def superpose(self, other: 'ChronoGraph') -> 'ChronoGraph':
        """Operador ⊕: fusão de dois grafos em superposição"""
        new = ChronoGraph()

        # União de nós com estados em superposição
        all_nodes = {**self.nodes, **other.nodes}
        for node_id, node in all_nodes.items():
            # Se existe em ambos, cria superposição
            if node_id in self.nodes and node_id in other.nodes:
                super_node = ChronoNode(
                    node_id=f"super_{node_id}",
                    kind='state',
                    value=(self.nodes[node_id].value, other.nodes[node_id].value),
                    quantum_state=QuantumState.SUPERPOSITION
                )
                new.add_node(super_node)
            else:
                new.add_node(node)

        # União de arestas
        for u, v, data in self.graph.edges(data=True):
            new.add_connection(u, v, data.get('relation'))
        for u, v, data in other.graph.edges(data=True):
            new.add_connection(u, v, data.get('relation'))

        # Arestas de interferência (novas conexões)
        for n1 in self.nodes.values():
            for n2 in other.nodes.values():
                if n1.kind == n2.kind:  # Ressonância por tipo
                    new.add_connection(
                        n1.node_id, n2.node_id, 'interferes_with'
                    )

        return new

    def collapse(self, input_data: Optional[Dict[str, Any]] = None,
                 max_universes: int = 100) -> List[Universe]:
        """
        Colapso do grafo: execução quântica-simbólica

        Implementa múltiplos universos paralelos que colapsam
        conforme observações (inputs) são feitas.
        """
        # Universo inicial (superposição total)
        root = Universe("U0")

        # Aplica inputs iniciais (observações externas)
        if input_data:
            for node_id, value in input_data.items():
                if node_id in self.nodes:
                    root.collapse_node(node_id, value)

        # Fila de processamento de superposições
        universes = [root]
        processed = []

        while universes and len(universes) + len(processed) < max_universes:
            current = universes.pop(0)

            # Encontra nós não colapsados
            unresolved = [
                nid for nid in self.nodes.keys()
                if nid not in current.state
            ]

            if not unresolved:
                processed.append(current)
                continue

            # Seleciona próximo nó para colapsar (heurística: órbitas antes de núcleos)
            next_node = min(unresolved,
                          key=lambda n: self.nodes[n].context.get('orbit_order', 999))

            node = self.nodes[next_node]

            # Determina possibilidades de colapso
            possibilities = self._get_possibilities(node, current)

            if len(possibilities) == 1:
                # Colapso determinístico
                current.collapse_node(next_node, possibilities[0])
                universes.append(current)
            else:
                # Ramificação em múltiplos universos
                branches = current.branch(next_node, possibilities)
                universes.extend(branches)

        # Retorna universos processados, ordenados por probabilidade
        return sorted(processed, key=lambda u: u.probability, reverse=True)

    def _get_possibilities(self, node: ChronoNode, universe: Universe) -> List[Any]:
        """Determina valores possíveis para um nó dado o contexto atual"""
        # Se tem valor fixo, apenas ele
        if node.value is not None and node.quantum_state != QuantumState.SUPERPOSITION:
            return [node.value]

        # Se é superposição, expande
        if isinstance(node.value, tuple):
            return list(node.value)

        # Se é operador, aplica aos valores de entrada
        if node.kind == 'orbit' and 'operator' in node.context:
            op = node.context['operator']
            # Busca operandos nos nós conectados
            operands = []
            for pred in self.graph.predecessors(node.node_id):
                if pred in universe.state:
                    operands.append(universe.state[pred])

            if len(operands) >= 2 and op in self.operadores:
                result = self.operadores[op](operands[0], operands[1])
                return [result] if result is not None else [None]

        # Caso geral: valor indefinido ou variável
        return [node.value, None, "undefined"]

    def extract_result(self, universe: Universe, projection: str = 'nucleus') -> Any:
        """
        Extrai resultado linear de um universo colapsado
        (necessário para interface com mundo externo)
        """
        if projection == 'nucleus':
            # Retorna valor do núcleo do primeiro sistema
            nucleus = [n for n in universe.state.keys() if 'nucleus' in n]
            return universe.state.get(nucleus[0]) if nucleus else None

        elif projection == 'all':
            # Retorna dicionário completo (perde holismo)
            return universe.state

        elif projection == 'probability':
            # Retorna probabilidade do universo
            return universe.probability

        return None

# Demonstração de execução
def demo_collapse():
    """Demonstra o motor de colapso"""
    cg = ChronoGraph()

    # Constrói: ◉5 ⊕ ○+3 → ◉8
    cg.add_node(ChronoNode('n1', 'core', 5))
    cg.add_node(ChronoNode('o1', 'orbit', None,
                          context={'operator': 'add', 'operand': 3}))
    cg.add_node(ChronoNode('n2', 'core', None))

    cg.add_connection('n1', 'o1', 'input')
    cg.add_connection('o1', 'n2', 'output')

    # Colapso
    universes = cg.collapse({'n1': 5})

    print(f"Gerados {len(universes)} universos:")
    for i, u in enumerate(universes[:5]):  # Top 5
        result = cg.extract_result(u, 'nucleus')
        print(f"  U{i}: {u.history} → resultado={result}, P={u.probability:.3f}")

    return universes
