"""
Arkhe Entanglement Coherence: C_global via GHZ/W states
"""
import numpy as np
import qutip as qt
from qutip import (
    Qobj, tensor, basis, sigmax, sigmay, sigmaz,
    ket2dm, ptrace as partial_trace, entropy_vn, qeye
)
from typing import List, Tuple, Literal, Dict
from .core import ArkheQobj, compute_local_coherence, ArkheHypergraph, QuantumHandover

class ArkheEntangledState:
    """
    Estado emaranhado multi-partido com métricas Arkhe(n).

    Suporta estados GHZ (maximalmente emaranhados) e W (robustos à perda)
    como bases para C_global emergente.
    """

    def __init__(self, n_qubits: int, state_type: Literal['GHZ', 'W'] = 'GHZ'):
        self.n = n_qubits
        self.type = state_type
        self.state = self._create_state()
        self.nodes = self._decompose_to_nodes()

    def _create_state(self) -> Qobj:
        """Cria estado emaranhado padrão."""
        if self.type == 'GHZ':
            # |GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2
            psi = (tensor([basis(2,0)]*self.n) + tensor([basis(2,1)]*self.n)).unit()
        elif self.type == 'W':
            # |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n
            terms = []
            for i in range(self.n):
                ket = [basis(2,0)]*self.n
                ket[i] = basis(2,1)
                terms.append(tensor(ket))
            psi = sum(terms).unit()
        else:
            raise ValueError(f"Unknown state_type: {self.type}. Supported types are 'GHZ' and 'W'.")

        return ket2dm(psi) if self.type == 'mixed' else psi

    def _decompose_to_nodes(self) -> List[ArkheQobj]:
        """Decompõe estado global em nós Arkhe com coerência local."""
        nodes = []
        for i in range(self.n):
            # Traço parcial para estado local: QuTiP ptrace mantém os índices fornecidos
            rho_i = partial_trace(self.state, [i])
            node = ArkheQobj(rho_i, node_id=f"entangled_node_{i}")

            # Coerência local de estado emaranhado é BAIXA (mistura máxima)
            # Para GHZ: cada qubit individual está em estado misto maximamente incoerente
            node.coherence = compute_local_coherence(rho_i)  # ≈ 0 para GHZ puro
            nodes.append(node)

        return nodes

    def compute_global_coherence(self) -> float:
        """
        C_global para estado emaranhado.

        Para estados GHZ puros: C_global = 1 (pureza global)
        Para estados W puros: C_global = 1
        Apesar de C_local ≈ 0 para todos os nós.
        """
        if self.state.isket:
            return 1.0  # Estado puro global

        # Para estados mistos, usar pureza global
        return np.real((self.state * self.state).tr())

    def compute_entanglement_entropy(self, bipartition: Tuple[List[int], List[int]]) -> float:
        """
        Entropia de emaranhamento para bipartição específica.
        Usada para verificar "monogamia" e estrutura de correlações.
        """
        set_a, set_b = bipartition
        # Verificar se bipartição é válida
        all_indices = set_a + set_b
        if sorted(all_indices) != list(range(self.n)):
            raise ValueError("Bipartição deve cobrir todos os índices")

        # Traço parcial sobre B
        rho_a = partial_trace(self.state, set_b)
        return entropy_vn(rho_a)

    def verify_emergence(self) -> Dict:
        """
        Verifica condição de emergência Arkhe(n): C_global > max(C_local).
        """
        c_locals = [n.coherence for n in self.nodes]
        c_global = self.compute_global_coherence()

        return {
            'c_global': c_global,
            'max_c_local': max(c_locals) if c_locals else 0,
            'mean_c_local': np.mean(c_locals) if c_locals else 0,
            'emergence_achieved': c_global > max(c_locals) if c_locals else True,
            'emergence_ratio': c_global / (max(c_locals) + 1e-10) if c_locals else 1.0,
            'entanglement_depth': self._compute_entanglement_depth()
        }

    def _compute_entanglement_depth(self) -> int:
        """
        Profundidade de emaranhamento: número mínimo de partículas emaranhadas.
        Para GHZ: n (emaranhamento genuíno n-partido)
        Para W: n (mas estrutura diferente de correlações)
        """
        # Simplificação: retornar n para estados puros emaranhados
        return self.n if self.compute_global_coherence() > 0.99 else 1


def create_arkhe_hypergraph_from_entangled(
    entangled_state: ArkheEntangledState,
    hypergraph_name: str = "Entangled Arkhe Graph"
) -> ArkheHypergraph:
    """
    Cria hipergrafo Arkhe a partir de estado emaranhado.

    Os handovers são **virtuais** — representam correlações quânticas
    não-locais que não podem ser simulados por interações locais clássicas.
    """
    hg = ArkheHypergraph(hypergraph_name)

    # Adicionar nós
    for node in entangled_state.nodes:
        hg.add_node(node)

    # Criar handovers "emaranhados" — representam correlações não-locais
    for i in range(entangled_state.n):
        for j in range(i+1, entangled_state.n):
            # Handover "fantasma" — marca correlação quântica
            h = QuantumHandover(
                handover_id=f"entanglement_{i}_{j}",
                source=entangled_state.nodes[i],
                target=entangled_state.nodes[j],
                operator=None,  # Não é operação local!
                metadata={
                    'type': 'entanglement_correlation',
                    'correlation_strength': _compute_correlation(
                        entangled_state.state, i, j
                    ),
                    'non_local': True
                }
            )
            hg.add_handover(h)

    # C_global do hipergrafo é coerência do estado global
    hg.global_coherence = entangled_state.compute_global_coherence()

    return hg


def _compute_correlation(rho: Qobj, i: int, j: int) -> float:
    """Computa correlação ⟨σ_z ⊗ σ_z⟩ entre qubits i e j."""
    if rho.isket:
        n = int(np.log2(rho.shape[0]))
    else:
        n = int(np.log2(rho.shape[0]))

    # Construir operador σ_z(i) ⊗ σ_z(j) no espaço total
    ops = [qeye(2)] * n
    ops[i] = sigmaz()
    ops[j] = sigmaz()
    corr_op = tensor(ops)

    if rho.isket:
        return np.real(qt.expect(corr_op, rho))
    else:
        return np.real((rho * corr_op).tr())


class EmergencePhaseTransition:
    """
    Simula transição de fase: separável → emaranhado → Arkhe-emergente.

    Modelo: estado misto ρ(p) = (1-p)|GHZ⟩⟨GHZ| + p·I/2^n
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.ghz = ArkheEntangledState(n_qubits, 'GHZ')
        self.max_mixed = Qobj(np.eye(2**n_qubits)/(2**n_qubits),
                              dims=[[2]*n_qubits, [2]*n_qubits])

    def mixed_state(self, p_noise: float) -> Qobj:
        """Estado misto com ruído p."""
        rho_ghz = ket2dm(self.ghz.state) if self.ghz.state.isket else self.ghz.state
        return (1 - p_noise) * rho_ghz + p_noise * self.max_mixed

    def scan_coherence(self, p_range: np.ndarray) -> Dict:
        """
        Scan de C_global vs C_local vs p_noise.

        Identifica "ponto crítico" onde emergência é perdida.
        """
        results = {'p': [], 'c_global': [], 'max_c_local': [], 'emergence': []}

        for p in p_range:
            rho = self.mixed_state(p)

            # C_global
            c_g = np.real((rho * rho).tr())

            # C_locals (média)
            c_locals = []
            for i in range(self.n):
                rho_i = partial_trace(rho, [i])
                c_locals.append(compute_local_coherence(rho_i))

            results['p'].append(p)
            results['c_global'].append(c_g)
            results['max_c_local'].append(max(c_locals))
            results['emergence'].append(c_g > max(c_locals))

        return results
