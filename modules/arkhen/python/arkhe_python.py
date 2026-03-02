# arkhe_python.py
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable
import ctypes
import pathlib
import os

# Carregar biblioteca C++ (assumindo que será compilada como libarkhe_field.so)
lib_path = pathlib.Path(__file__).parent.parent / "cpp" / "libarkhe_field.so"
if os.path.exists(lib_path):
    _lib = ctypes.CDLL(str(lib_path))
else:
    _lib = None

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI

@dataclass
class NodeConfig:
    """Configuração de nó Arkhe(n)"""
    dimension: int = 10
    coupling: float = PHI_INV  # λ = φ
    coherence_threshold: float = 0.5
    max_iterations: int = 99  # Loop cristalino

class ArkheField:
    """
    Campo Ψ 10D em Python (wrapper sobre C++).

    Interface de alto nível para prototipagem rápida
    antes de compilação para Rust/C++.
    """

    def __init__(self, config: NodeConfig = None):
        self.config = config or NodeConfig()
        self._field = np.zeros((1024, self.config.dimension), dtype=np.complex64)
        self._coherence = np.full(1024, 0.5)
        self._timestamp = 0

    def initialize_maximum_entropy(self):
        """Estado H=2.0 (máxima entropia com coerência 0.5)"""
        self._field = np.random.randn(1024, self.config.dimension) + \
                      1j * np.random.randn(1024, self.config.dimension)
        norm = np.linalg.norm(self._field, axis=1, keepdims=True)
        self._field = np.where(norm > 0, self._field / norm, self._field)
        self._coherence = np.full(1024, 0.5)

    def evolve(self, dt: float = 0.01, handovers: List[dict] = None):
        """
        Evolução do campo: ∂Ψ/∂t = ∇²Ψ - λ|Ψ|²Ψ + J

        Args:
            dt: passo temporal
            handovers: lista de handovers (fonte J)
        """
        # Laplaciano (difusão)
        laplacian = self._compute_laplacian()

        # Não-linearidade crítica
        psi_squared = np.abs(self._field)**2
        nonlinear = self.config.coupling * psi_squared * self._field

        # Fonte (handovers)
        source = np.zeros_like(self._field)
        if handovers:
            for h in handovers:
                source += self._integrate_handover(h)

        # Evolução
        self._field += dt * (laplacian - nonlinear + source)

        # Normalizar
        norm = np.linalg.norm(self._field, axis=1, keepdims=True)
        self._field = np.where(norm > 0, self._field / norm, self._field)

        # Atualizar coerência
        self._update_coherence()

        self._timestamp += 1

        return self._coherence.mean()

    def _compute_laplacian(self) -> np.ndarray:
        """Laplaciano 10D com condições periódicas"""
        laplacian = np.zeros_like(self._field)
        for d in range(self.config.dimension):
            # Diferenças finitas (pode ser otimizado com FFT)
            laplacian[:, d] = np.roll(self._field[:, d], 1) + \
                              np.roll(self._field[:, d], -1) - \
                              2 * self._field[:, d]
        return laplacian

    def _integrate_handover(self, handover: dict) -> np.ndarray:
        """Integra dados de handover no campo"""
        return np.zeros_like(self._field)

    def _update_coherence(self):
        """ρ = 1 - H/H_max (coerência como inverso da entropia)"""
        for i in range(1024):
            probs = np.abs(self._field[i])**2
            probs = probs / (probs.sum() + 1e-10)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(self.config.dimension)
            self._coherence[i] = 1 - (entropy / max_entropy)

    def get_critical_nodes(self) -> np.ndarray:
        """Nós em ρ > φ (regime cristalino)"""
        return np.where(self._coherence > PHI_INV)[0]

    def render_visual(self) -> np.ndarray:
        """
        Gerar imagem para shader GLSL.

        Retorna array (H, W, 3) com visualização do campo.
        """
        # Projeção 10D → 3D (similar ao shader)
        projected = np.zeros((1024, 3))
        for i in range(1024):
            r = np.linalg.norm(self._field[i, :2])
            projected[i] = [
                np.log2(r + 1e-6),
                -self._field[i, 2].real / (r + 1e-6) - 0.8,
                np.arctan2(self._field[i, 0].real * 0.08, self._field[i, 1].real)
            ]

        return projected

class ArkheNetwork:
    """Rede de nós Arkhe(n) com handovers"""

    def __init__(self, n_nodes: int = 16):
        self.nodes = [ArkheField() for _ in range(n_nodes)]
        self.ledger = []  # Lista de handovers registrados

    def connect_nodes(self, i: int, j: int, strength: float = PHI_INV):
        """Criar Noether Channel entre nós"""
        pass

    def simulate_step(self):
        """Um passo de simulação da rede completa"""
        handovers = []
        for i, node in enumerate(self.nodes):
            critical = node.get_critical_nodes()
            for j in critical:
                handover = self._create_handover(i, j)
                handovers.append(handover)

        for h in handovers:
            target_node = self.nodes[h['target_node']]
            target_node.evolve(handovers=[h['data']])

        self.ledger.extend(handovers)

        global_coherence = np.mean([n._coherence.mean() for n in self.nodes])

        return global_coherence

    def _create_handover(self, source_idx, node_idx):
        return {
            'source_node': source_idx,
            'target_node': (source_idx + 1) % len(self.nodes),
            'data': np.zeros(10)
        }

# API Unificada (Exposta como no pedido)
def Node(coupling=PHI_INV, dimension=10):
    config = NodeConfig(coupling=coupling, dimension=dimension)
    return ArkheField(config)
