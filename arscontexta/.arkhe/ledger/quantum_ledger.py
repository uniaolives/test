# .arkhe/ledger/quantum_ledger.py
import time
import networkx as nx
import numpy as np

class QuantumLedger:
    """
    Registro de emaranhamento de longo prazo entre nós do hipergrafo.
    Mapeia handovers de alta Φ para conexões EPR persistentes.
    """
    def __init__(self):
        self.entanglement_graph = nx.Graph()  # Nós são qubits, arestas são EPRs

    def record_handover(self, source_id: str, target_id: str, phi_score: float):
        """
        Registra um handover. Se phi_score >= 1.618 (φ²), cria/reforça emaranhamento.
        """
        # φ² como limiar de "eternidade" ou ressonância áurea
        if phi_score >= 1.618:
            if self.entanglement_graph.has_edge(source_id, target_id):
                # Reforçar emaranhamento existente
                self.entanglement_graph[source_id][target_id]['weight'] = \
                    min(1.0, self.entanglement_graph[source_id][target_id]['weight'] + 0.05)
            else:
                self.entanglement_graph.add_edge(source_id, target_id,
                                               weight=0.5, # Fidelidade inicial
                                               timestamp=time.time())
            print(f"🔗 [QUANTUM LEDGER] EPR Pair established/reinforced between {source_id} and {target_id}")

    def query_fidelity(self, node_a: str, node_b: str) -> float:
        """
        Verifica a fidelidade de emaranhamento entre dois nós.
        A fidelidade decai com o número de hops (quantum swaps).
        """
        if node_a not in self.entanglement_graph or node_b not in self.entanglement_graph:
            return 0.0

        if not nx.has_path(self.entanglement_graph, node_a, node_b):
            return 0.0

        path = nx.shortest_path(self.entanglement_graph, node_a, node_b)
        # Fidelidade decai multiplicativamente ao longo do caminho
        fidelity = 1.0
        for i in range(len(path) - 1):
            fidelity *= self.entanglement_graph[path[i]][path[i+1]]['weight']

        return float(fidelity)

    def get_entanglement_density(self) -> float:
        """Retorna a densidade de conexões EPR na rede."""
        num_nodes = self.entanglement_graph.number_of_nodes()
        if num_nodes < 2: return 0.0
        return float(self.entanglement_graph.number_of_edges() / (num_nodes * (num_nodes - 1) / 2))
