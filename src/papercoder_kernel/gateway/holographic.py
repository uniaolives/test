from ..multivac.multivac_substrate import MultivacSubstrate, ComputeNode
from typing import Dict, Any, Optional

class HolographicInterface:
    """
    Implementação matemática da interface holográfica de 1983.
    """

    def __init__(self, substrate: MultivacSubstrate):
        self.substrate = substrate
        self.absolute = self.substrate  # O "Absolute" = H em sua totalidade

    def principle_of_holography(self, node_id: str) -> Dict[str, Any]:
        """
        Princípio: cada parte contém informação do todo.

        No hipergrafo: cada nó v tem acesso potencial a todo H
        via caminhos de arestas (handovers).
        """
        if node_id not in self.substrate.nodes:
            return {"error": "Node not found"}

        node = self.substrate.nodes[node_id]

        # Informação local
        local_info = node.node_type

        # Informação global acessível (dentro do horizonte de coerência)
        # max_depth proporcional à coerência
        max_depth = int(node.coherence * 10)
        accessible = self.substrate.bfs_from_node(node_id, max_depth=max_depth)

        # Razão holográfica: quanto do todo está em cada parte
        total_nodes = len(self.substrate.nodes)
        holographic_ratio = len(accessible) / total_nodes if total_nodes > 0 else 0

        return {
            'local': local_info,
            'accessible_global_count': len(accessible),
            'ratio': holographic_ratio,
            'principle_satisfied': holographic_ratio > 0
        }

    def non_local_perception(self, viewer_id: str, target_id: str) -> Dict[str, Any]:
        """
        Remote viewing como acesso a nó distante no hipergrafo.
        """
        if viewer_id not in self.substrate.nodes or target_id not in self.substrate.nodes:
             return {'perceived': None, 'reason': 'Node not found', 'accuracy': 0}

        viewer = self.substrate.nodes[viewer_id]
        path = self.substrate.find_path(viewer_id, target_id)

        if path and viewer.coherence > 0.8:  # Focus 21+
            # Percepção não-local possível
            # A acurácia decai com a distância (comprimento do caminho)
            accuracy = viewer.coherence * (1.0 - (len(path) - 1) / 100.0)
            accuracy = max(0, accuracy)

            return {
                'perceived': self.substrate.nodes[target_id].node_type,
                'accuracy': accuracy,
                'mechanism': 'hypergraph_path_access',
                'boundary_conditions': 'C_viewer > 0.8, path exists'
            }
        else:
            return {
                'perceived': None,
                'reason': 'Insufficient coherence or no path',
                'accuracy': 0
            }
