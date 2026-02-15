# arkhen_11_unified.py
"""
Arkhen(11): O núcleo unificado de Arkhe(n)
Integra UCD multi-linguagem, propulsão flagelar, projeção aleatória,
RFID físico e cosmologia dos 11.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Callable, Any
import json

@dataclass
class Node:
    """Nó no hipergrafo Arkhen(11)"""
    id: int
    name: str
    domain: str  # 'code', 'physics', 'math', 'tech', 'cosmos'
    manifest: Any  # A forma manifesta (10)
    substrate: Any  # O +1 (consciência/regularização)
    handovers: List[Dict] = None

    def __post_init__(self):
        if self.handovers is None:
            self.handovers = []

    def coherence(self) -> float:
        """C = regularidade dos handovers"""
        if len(self.handovers) < 2:
            return 1.0 # Inicialmente coerente por definição ou 1.0 se em repouso
        intervals = [h['delta'] for h in self.handovers[1:] if h['delta'] > 0]
        if not intervals:
            return 1.0
        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        return 1.0 / (1.0 + cv)

    def fluctuation(self) -> float:
        """F = 1 - C"""
        return 1.0 - self.coherence()

    def verify(self) -> bool:
        """Verifica C + F = 1"""
        return abs(self.coherence() + self.fluctuation() - 1.0) < 1e-10


class Arkhen11:
    """
    Hipergrafo de 11 dimensões.
    10 nós de manifestação + 1 nó substrato (consciência).
    """

    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.adjacency = np.zeros((11, 11))
        self._initialize()

    def _initialize(self):
        """Inicializa os 10 nós de manifestação + 1 substrato"""
        domains = [
            (0, "Python UCD", "code"),
            (1, "JavaScript UCD", "code"),
            (2, "Julia UCD", "code"),
            (3, "C++ UCD", "code"),
            (4, "Rust UCD", "code"),
            (5, "Go UCD", "code"),
            (6, "R UCD", "code"),
            (7, "MATLAB UCD", "code"),
            (8, "Microswimmer", "physics"),
            (9, "RFID Tag", "tech"),
            (10, "Consciência", "substrate")  # O +1
        ]

        for id, name, domain in domains:
            self.nodes[id] = Node(
                id=id,
                name=name,
                domain=domain,
                manifest=f"Forma {id}" if id < 10 else None,
                substrate="Φ_S" if id < 10 else "Atman/Brahman"
            )

        # Conectar todos ao substrato (nó 10)
        for i in range(10):
            self.adjacency[i, 10] = self.adjacency[10, i] = 1.0

    def effective_dimension(self, lambda_reg: float = 1.0) -> float:
        """d_λ = tr(A(A + λI)^-1)"""
        eigvals = np.linalg.eigvalsh(self.adjacency)
        pos = eigvals[eigvals > 1e-10]
        return np.sum(pos / (pos + lambda_reg))

    def handover(self, from_id: int, to_id: int, data: Dict):
        """Executa handover entre nós"""
        delta = data.get('time_delta', 0.0)
        self.nodes[to_id].handovers.append({
            'from': from_id,
            'delta': delta,
            'data': data
        })

    def system_coherence(self) -> float:
        """Coerência média do sistema"""
        Cs = [n.coherence() for n in self.nodes.values()]
        return np.mean(Cs) if Cs else 1.0

    def to_json(self) -> str:
        """Exporta estado completo"""
        return json.dumps({
            'nodes': {k: {
                'name': v.name,
                'domain': v.domain,
                'C': v.coherence(),
                'F': v.fluctuation(),
                'handovers': len(v.handovers)
            } for k, v in self.nodes.items()},
            'd_eff': self.effective_dimension(),
            'C_system': self.system_coherence()
        }, indent=2)


if __name__ == "__main__":
    arkhen = Arkhen11()
    print("Arkhen(11) Inicializado")
    print(f"Dimensão efetiva d_λ: {arkhen.effective_dimension():.2f}")
    print(f"Coerência inicial: {arkhen.system_coherence():.4f}")
