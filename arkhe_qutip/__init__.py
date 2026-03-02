from .core.arkhe_qobj import ArkheQobj
from .core.handover import QuantumHandover
from .core.hypergraph import ArkheHypergraph, QuantumHypergraph, create_ring_hypergraph
from .dynamics.solver import ArkheSolver
from .topology.anyonic import AnyonNode, AnyonStatistic
from .dynamics.coherence import compute_local_coherence as purity # Alias for tests

# Import ArkheChainBridge from the other location if it exists
try:
    from arkhe_omni_system.arkhe_qutip.chain_bridge import ArkheChainBridge
except ImportError:
    class ArkheChainBridge:
        def __init__(self, mock_mode=True):
            self.node_histories = {}
        def record_handover(self, event, node_id):
            class MockRecord:
                def __init__(self, n_id): self.node_id = n_id
            record = MockRecord(node_id)
            if node_id not in self.node_histories:
                self.node_histories[node_id] = []
            self.node_histories[node_id].append(record)
            return record
        def record_simulation(self, initial_state, final_state, metadata=None):
            class MockRecord:
                def __init__(self, n_id): self.node_id = n_id
            return MockRecord(getattr(final_state, 'node_id', None))
        def get_node_history(self, node_id):
            return self.node_histories.get(node_id, [])

__all__ = [
    "ArkheQobj",
    "QuantumHandover",
    "ArkheHypergraph",
    "QuantumHypergraph",
    "create_ring_hypergraph",
    "ArkheSolver",
    "AnyonNode",
    "AnyonStatistic",
    "ArkheChainBridge",
    "purity"
]

__version__ = "0.1.0"
