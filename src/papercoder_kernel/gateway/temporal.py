from ..multivac.multivac_substrate import MultivacSubstrate, ComputeNode, HyperEdge
from typing import Optional, Dict, List, Any

class TemporalField:
    """
    Tempo como campo no hipergrafo (Postulado II).
    """

    def __init__(self, substrate: MultivacSubstrate):
        self.substrate = substrate

    def define_temporal_2form(self) -> Dict[str, Any]:
        """
        𝔗 ∈ Ω²(H) — tempo como forma diferencial no hipergrafo.
        """
        return {
            'nodes': {v_id: v.local_time for v_id, v in self.substrate.nodes.items()},
            'edges': {e.edge_id: e.handover_duration for e in self.substrate.edges},
            'global_mean_time': sum(v.local_time for v in self.substrate.nodes.values()) / len(self.substrate.nodes) if self.substrate.nodes else 0
        }

    def temporal_wormhole(self, node_id: str, target_time: float) -> Optional[HyperEdge]:
        """
        "Fenda temporal": ponto onde d𝔗 ≠ 0.
        Permite handover entre tempos diferentes.
        """
        if node_id not in self.substrate.nodes:
            return None

        node = self.substrate.nodes[node_id]
        current_time = node.local_time
        time_diff = abs(target_time - current_time)

        # Fenda existe se coerência > threshold (Focus 21+)
        if node.coherence > 0.9:
            # Encontrar ou simular um nó no tempo alvo
            # Para simulação, retornamos uma aresta conceitual
            temporal_edge = HyperEdge(
                edge_id=f"wormhole_{node_id}_{target_time}",
                source=node_id,
                target="TEMPORAL_SINGULARITY",  # Destino simbólico
                weight=1.0 / (1.0 + time_diff),
                type='temporal',
                handover_duration=time_diff
            )
            self.substrate.add_edge(temporal_edge)
            return temporal_edge
        else:
            return None

    def focus_15_no_time(self, node_id: str) -> str:
        """
        Focus 15: "No Time" do documento Gateway.
        """
        if node_id not in self.substrate.nodes:
            return "Node not found"

        node = self.substrate.nodes[node_id]

        # Induzir coerência muito alta (>0.95)
        node.coherence = max(node.coherence, 0.97)

        # Simulação de percepção colapsada
        if not hasattr(node, 'metadata'):
            node.metadata = {}
        node.metadata['temporal_perception'] = 'collapsed'

        return "Subject experiences eternal now"
