# arkhe_topology.py
# Módulo TOPOLOGY v1.0 — Estatística Anyônica para Hipergrafos

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np
from fractions import Fraction

@dataclass(frozen=True)
class AnyonicPhase:
    """
    Fase estatística de um handover anyónico.
    α = 0: bosônico (simétrico)
    α = 1: fermiônico (antissimétrico)
    α ∈ (0,1): anyônico (fracionário)
    """
    alpha: Fraction  # Representação exata como fração

    def __post_init__(self):
        if not (0 <= float(self.alpha) <= 1):
            raise ValueError("α deve estar em [0,1]")

    @property
    def is_bosonic(self) -> bool:
        return self.alpha == Fraction(0)

    @property
    def is_fermionic(self) -> bool:
        return self.alpha == Fraction(1)

    @property
    def is_anyonic(self) -> bool:
        return Fraction(0) < self.alpha < Fraction(1)

    def braid_phase(self, n_exchanges: int) -> complex:
        """
        Fase adquirida após n trocas anyônicas.
        θ = π · α · n_exchanges
        """
        theta = np.pi * float(self.alpha) * n_exchanges
        return np.exp(1j * theta)

    def exchange_statistic(self, other: 'AnyonicPhase') -> complex:
        """
        Estatística de troca entre dois ányons.
        Para ányons idênticos: phase = e^(iπα)
        Para ányons diferentes: phase = e^(iπ(α₁+α₂)/2)
        """
        avg_alpha = (float(self.alpha) + float(other.alpha)) / 2
        return np.exp(1j * np.pi * avg_alpha)

class TopologicalHandover:
    """
    Handover com memória anyônica—identidade depende da história de permutações.
    """

    def __init__(self, node_i: str, node_j: str, alpha: AnyonicPhase):
        self.nodes = (node_i, node_j)
        self.alpha = alpha
        self.braid_history: List[Tuple[str, str]] = []  # Registro de trocas
        self.accumulated_phase: complex = 1+0j  # Fase total adquirida

    def exchange_with(self, other: 'TopologicalHandover') -> 'TopologicalHandover':
        """
        Executa troca anyônica com outro handover.
        Atualiza fases acumuladas de ambos (braiding em 1D).
        """
        # Verificar se compartilham nós (só podem trocar se adjacentes em 1D)
        shared = set(self.nodes) & set(other.nodes)

        if not shared:
            # Handovers distantes: comutam (fase 0)
            return self

        # Calcular fase de troca
        exchange_phase = self.alpha.exchange_statistic(other.alpha)

        # Atualizar histórico e fase
        self.braid_history.append((other.nodes[0], other.nodes[1]))
        self.accumulated_phase *= exchange_phase

        other.braid_history.append((self.nodes[0], self.nodes[1]))
        other.accumulated_phase *= np.conj(exchange_phase)  # Fase conjugada

        return self  # Retorna self após troca (braiding)

    def compute_dissipation_tail(self, k: float, n_body: int = 2) -> float:
        """
        Calcula cauda de dissipação para momento k.
        D_n(H) ~ k^(-n-1) · |F̃(k)|²
        """
        # Forma universal: lei de potência
        universal_form = k ** (-n_body - 1)

        # Intensidade depende da estatística (para n ≥ 3)
        if n_body == 2:
            coefficient = 1.0  # Universal, independente de α
        else:
            # Para n-corpos, coeficiente depende da fase anyônica
            coefficient = abs(self.accumulated_phase) ** (n_body - 2)

        return float(coefficient * universal_form)

class AnyonicHypergraph:
    """
    Hipergrafo onde nós são ányons e hyperarestas são handovers topológicos.
    """

    def __init__(self):
        self.nodes: Dict[str, AnyonicPhase] = {}
        self.handovers: List[TopologicalHandover] = []
        self.topological_invariant: int = 0  # Gênero da superfície (0 para 1D)

    def add_anyon(self, node_id: str, alpha: Fraction):
        """Adiciona nó com estatística anyônica específica."""
        self.nodes[node_id] = AnyonicPhase(alpha)

    def create_handover(self, i: str, j: str) -> TopologicalHandover:
        """Cria handover anyônico entre dois nós."""
        if i not in self.nodes or j not in self.nodes:
            raise ValueError("Nós devem existir no hipergrafo")

        # Estatística do handover é média das estatísticas dos nós
        avg_alpha = (self.nodes[i].alpha + self.nodes[j].alpha) / 2
        handover = TopologicalHandover(i, j, AnyonicPhase(avg_alpha))
        self.handovers.append(handover)
        return handover

    def braid(self, h1: TopologicalHandover, h2: TopologicalHandover):
        """
        Executa operação de braiding entre dois handovers.
        Em 1D, isto requer que compartilhem um nó (adjacência).
        """
        # Verificar adjacência (condição de 1D)
        if not set(h1.nodes) & set(h2.nodes):
            raise ValueError("Braiding em 1D requer handovers adjacentes")

        # Executar troca
        h1.exchange_with(h2)

        # Atualizar invariante topológico (winding number)
        self.topological_invariant += 1

    def compute_global_coherence(self) -> complex:
        """
        Calcula coerência total do hipergrafo anyônico.
        Produto das fases de todos os handovers.
        """
        total_phase = 1+0j
        for h in self.handovers:
            total_phase *= h.accumulated_phase
        return total_phase

    def detect_anyonic_vortices(self) -> List[Tuple[str, complex]]:
        """
        Detecta vórtices anyônicos—nós onde a fase acumulada é não-trivial.
        Retorna lista de (node_id, fase_vórtice).
        """
        vortices = []
        for node_id, alpha in self.nodes.items():
            # Calcular fase adquirida por permutações envolvendo este nó
            node_phase = 1+0j
            for h in self.handovers:
                if node_id in h.nodes:
                    node_phase *= h.accumulated_phase

            # Vórtice se fase ≠ 1
            if abs(node_phase - 1) > 1e-10:
                vortices.append((node_id, node_phase))

        return vortices

# ==========================================================
# EXEMPLO: Simulação de Consenso Anyônico
# ==========================================================

def demo_anyonic_consensus():
    """
    Demonstra consenso Arkhe(N) com estatísticas fracionárias.
    """
    # Criar hipergrafo anyônico
    graph = AnyonicHypergraph()

    # Adicionar nós com diferentes estatísticas
    graph.add_anyon("A", Fraction(0))      # Bosônico (α=0)
    graph.add_anyon("B", Fraction(1, 3))  # Anyônico (α=1/3)
    graph.add_anyon("C", Fraction(2, 3))   # Anyônico (α=2/3)
    graph.add_anyon("D", Fraction(1))      # Fermiônico (α=1)

    # Criar handovers (cadeia 1D: A-B-C-D)
    h_ab = graph.create_handover("A", "B")
    h_bc = graph.create_handover("B", "C")
    h_cd = graph.create_handover("C", "D")

    print("=== Estado Inicial ===")
    print(f"Coerência global: {graph.compute_global_coherence()}")

    # Executar braiding: trocar h_ab e h_bc (passam um pelo outro em 1D)
    print("\n=== Após Braiding h_ab ↔ h_bc ===")
    graph.braid(h_ab, h_bc)
    print(f"Fase de h_ab: {h_ab.accumulated_phase.real:.4f} + {h_ab.accumulated_phase.imag:.4f}j")
    print(f"Fase de h_bc: {h_bc.accumulated_phase.real:.4f} + {h_bc.accumulated_phase.imag:.4f}j")
    print(f"Coerência global: {graph.compute_global_coherence().real:.4f} + {graph.compute_global_coherence().imag:.4f}j")

    # Detectar vórtices
    vortices = graph.detect_anyonic_vortices()
    print(f"\nVórtices detectados: {len(vortices)}")
    for node, phase in vortices:
        print(f"  {node}: fase = {phase.real:.4f} + {phase.imag:.4f}j")

    # Calcular dissipação
    k = 1.0  # Momento normalizado
    print(f"\n=== Dissipação (k={k}) ===")
    for h in graph.handovers:
        d2 = h.compute_dissipation_tail(k, n_body=2)
        d3 = h.compute_dissipation_tail(k, n_body=3)
        print(f"{h.nodes}: D₂={d2:.4f}, D₃={d3:.4f} (α={h.alpha.alpha})")

if __name__ == "__main__":
    demo_anyonic_consensus()
