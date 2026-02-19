import pytest
import numpy as np
import qutip as qt
from arkhe_qutip import (
    ArkheQobj,
    ArkheSolver,
    QuantumHypergraph,
    create_ring_hypergraph,
    ArkheChainBridge,
    purity
)

def test_arkhe_qobj_handover():
    from qutip_qip.operations import hadamard_transform
    psi = ArkheQobj(qt.basis(2, 0))  # |0>
    assert psi.coherence == 1.0

    psi = psi.handover(hadamard_transform(), {'type': 'superposition'})
    assert np.isclose(psi.coherence, 1.0)
    assert len(psi.history) == 1
    assert "superposition" in psi.history[0].metadata['type']

def test_quantum_hypergraph():
    hg = create_ring_hypergraph(5)
    assert hg.n_nodes == 5
    assert hg.n_hyperedges == 5
    assert hg.global_coherence == 1.0

    stats = hg.get_topology_stats()
    assert stats['n_nodes'] == 5
    assert stats['avg_degree'] == 2.0

def test_arkhe_solver():
    H = qt.sigmaz()
    c_ops = [0.1 * qt.sigmax()]
    solver = ArkheSolver(H, c_ops, phi_coupling=0.05)

    psi0 = ArkheQobj(qt.basis(2, 0))
    tlist = np.linspace(0, 1, 10)
    result = solver.solve(psi0, tlist, track_coherence=True)

    assert len(result.states) == 10
    assert len(result.coherence) == 10
    assert result.coherence[0] == 1.0
    assert hasattr(result, 'phi_trajectory')
    assert hasattr(result, 'coherence_trajectory')
    assert isinstance(result.arkhe_final_state, ArkheQobj)

    # Test history propagation
    psi0_with_history = psi0.handover(qt.sigmax(), {'type': 'initial-flip'})
    result2 = solver.solve(psi0_with_history, tlist, track_coherence=True)
    assert len(result2.arkhe_final_state.history) == 1
    assert result2.arkhe_final_state.node_id == psi0_with_history.node_id

def test_chain_bridge():
    bridge = ArkheChainBridge(mock_mode=False)
    psi = ArkheQobj(qt.basis(2, 0))
    psi = psi.handover(qt.sigmax(), {'type': 'bit-flip'})

    record = bridge.record_handover(psi.history[0], psi.node_id)
    assert record.node_id == psi.node_id
    assert len(bridge.get_node_history(psi.node_id)) == 1

    # Test record_simulation with new params
    sim_record = bridge.record_simulation(initial_state=psi, final_state=psi)
    assert sim_record.node_id == psi.node_id

def test_evolve_with_handover():
    psi0 = ArkheQobj(qt.basis(2, 0))
    H = qt.sigmaz()
    tlist = np.linspace(0, 10, 10)
    handovers = [
        (5.0, qt.sigmax(), {'type': 'mid-flip'})
    ]

    states, _ = psi0.evolve_with_handover(H, tlist, handovers)
    assert len(states) >= 10
    # The final state should have the handover history
    assert len(states[-1].history) == 1
