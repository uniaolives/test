# arkhe_qutip/__init__.py
"""
Arkhe-QuTiP: Quantum Hypergraph Toolbox
Extension of QuTiP for quantum hypergraph structures with Arkhe(N) coherence tracking and handover mechanics.
"""

from .core import ArkheQobj, ArkheSolver, HandoverEvent
from .hypergraph import QuantumHypergraph, Hyperedge, create_ring_hypergraph
from .coherence import (
    purity,
    von_neumann_entropy,
    coherence_l1,
    integrated_information,
    coherence_trajectory_analysis
)
from .visualization import plot_hypergraph, plot_coherence_trajectory
from .chain_bridge import ArkheChainBridge
from .fpga import FPGAQubitEmulator, ArkheFPGAMiner
from .network import ArkheNetworkNode, DistributedPoCConsensus, ArkheHypergraphServicer, serve_arkhe_node

__version__ = "1.2.0"

__version__ = "1.0.0"

__all__ = [
    "ArkheQobj",
    "ArkheSolver",
    "HandoverEvent",
    "QuantumHypergraph",
    "Hyperedge",
    "create_ring_hypergraph",
    "purity",
    "von_neumann_entropy",
    "coherence_l1",
    "integrated_information",
    "coherence_trajectory_analysis",
    "plot_hypergraph",
    "plot_coherence_trajectory",
    "ArkheChainBridge",
    "FPGAQubitEmulator",
    "ArkheFPGAMiner",
    "ArkheNetworkNode",
    "DistributedPoCConsensus",
    "ArkheHypergraphServicer",
    "serve_arkhe_node",
]
