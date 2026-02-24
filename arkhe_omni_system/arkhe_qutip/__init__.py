# arkhe_omni_system/arkhe_qutip/__init__.py
"""
Arkhe-QuTiP: Quantum Hypergraph Toolbox (Omni-System Edition)
"""

from .core import ArkheQobj, HandoverEvent, QuantumHandover, ArkheHypergraph, compute_local_coherence
from .solver import ArkheSolver
from .hypergraph import QuantumHypergraph, create_ring_hypergraph
from .entanglement import ArkheEntangledState, EmergencePhaseTransition, create_arkhe_hypergraph_from_entangled
from .lindblad import ArkheLindbladian, AIAuthorityBlocker, ConstitutionMonitor, CognitiveLoad
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
from .consensus import DistributedPoCConsensus
from .acoustic_time_crystal import AcousticTimeCrystal
from .server import ArkheHypergraphServicer, serve_arkhe_node

__version__ = "1.2.0"

__all__ = [
    "ArkheQobj",
    "ArkheSolver",
    "HandoverEvent",
    "QuantumHandover",
    "ArkheHypergraph",
    "compute_local_coherence",
    "QuantumHypergraph",
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
    "DistributedPoCConsensus",
    "AcousticTimeCrystal",
    "ArkheHypergraphServicer",
    "serve_arkhe_node",
    "ArkheEntangledState",
    "EmergencePhaseTransition",
    "create_arkhe_hypergraph_from_entangled",
    "ArkheLindbladian",
    "AIAuthorityBlocker",
    "ConstitutionMonitor",
    "CognitiveLoad",
]
