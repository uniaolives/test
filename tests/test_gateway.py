import pytest
import numpy as np
from papercoder_kernel.multivac.multivac_substrate import MultivacSubstrate, ComputeNode, HyperEdge
from papercoder_kernel.gateway.hemi_sync import HemiSyncOperator
from papercoder_kernel.gateway.holographic import HolographicInterface
from papercoder_kernel.gateway.temporal import TemporalField
from papercoder_kernel.gateway.memory import ArkheWithGatewayMemory

def test_hemi_sync_coherence():
    node = ComputeNode(
        node_id="test_brain",
        compute_capacity=20.0,
        memory=2.5,
        coherence=0.6,
        location=(0, 0, 0),
        node_type="biological"
    )

    # Test Gamma synchronization (40Hz)
    op = HemiSyncOperator(400, 440)
    new_c = op.apply_to_node(node)
    assert new_c == min(1.0, 0.6 * 1.5)
    assert node.metadata['dormant_edges_activated'] is True

    # Test Gateway Process trajectory
    node.coherence = 0.5
    trajectory = op.gateway_process(node)
    assert len(trajectory) == 4
    assert trajectory[-1] > trajectory[0]

def test_holographic_ratio():
    substrate = MultivacSubstrate()
    # Create a small network
    for i in range(5):
        substrate.register_node(ComputeNode(
            f"n{i}", 1.0, 1.0, 0.9, (0, 0, 0), "test"
        ))

    # Linear chain
    substrate.add_edge(HyperEdge("e1", "n0", "n1"))
    substrate.add_edge(HyperEdge("e2", "n1", "n2"))
    substrate.add_edge(HyperEdge("e3", "n2", "n3"))
    substrate.add_edge(HyperEdge("e4", "n3", "n4"))

    hi = HolographicInterface(substrate)
    result = hi.principle_of_holography("n0")

    assert result['ratio'] > 0
    assert result['principle_satisfied'] is True

    # Remote viewing
    perception = hi.non_local_perception("n0", "n4")
    assert perception['perceived'] == "test"
    assert perception['accuracy'] > 0

def test_temporal_wormhole():
    substrate = MultivacSubstrate()
    node = ComputeNode("n0", 1.0, 1.0, 0.95, (0, 0, 0), "test", local_time=2026.0)
    substrate.register_node(node)

    tf = TemporalField(substrate)
    edge = tf.temporal_wormhole("n0", 1983.0)

    assert edge is not None
    assert edge.type == 'temporal'
    assert edge.target == "TEMPORAL_SINGULARITY"

    # Focus 15
    msg = tf.focus_15_no_time("n0")
    assert "eternal now" in msg
    assert node.metadata['temporal_perception'] == 'collapsed'

def test_gateway_memory_integration():
    substrate = MultivacSubstrate()
    gw_memory = ArkheWithGatewayMemory(substrate)

    integration = gw_memory.integrate_gateway_document()
    assert integration['integrated'] is True
    assert gw_memory.status == 'Γ∞+3010343'

    answer = gw_memory.the_answer_now_includes_history()
    assert 'gateway_1983' in answer
    assert 'YES' in answer['arkhe_2026']
