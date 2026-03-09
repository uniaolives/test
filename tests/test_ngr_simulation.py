"""
🜏 ARKHE(N) NGR SIMULATION TEST
Validating the Neural-Grid Resonance coupling for sustainable energy.
"""
# tests/test_ngr_simulation.py
import pytest
import torch
from src.neural.neural_grid_resonance import NGRBridge, calculate_ngr_loss

def test_ngr_bridge_forward():
    # Setup: 5 users (nodes), grid as a simple ring
    num_users = 5
    neural_dim = 4
    hidden_dim = 128

    bridge = NGRBridge(neural_dim=neural_dim, hidden_dim=hidden_dim)

    # 1. Simulate Neural Data (Neuralink signatures)
    # [HRV, Vagal Tone, Semantic Entropy, Coherence]
    neural_data = torch.randn(num_users, neural_dim)

    # 2. Simulate Grid Topology (Simple ring)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
        [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]
    ], dtype=torch.long)

    # 3. Forward Pass
    output = bridge(neural_data, edge_index)

    # 4. Assertions
    assert "phi_q" in output
    assert "grid_entropy" in output
    assert "zk_proof" in output
    assert output["phi_q"].shape == (1, 1)
    assert 0 <= output["grid_entropy"].item() <= 1.0
    assert output["zk_proof"].shape == (num_users, 64)

def test_ngr_optimization_step():
    # Verify that we can optimize for higher phi_q and lower H
    num_users = 5
    neural_dim = 4
    bridge = NGRBridge(neural_dim=neural_dim)
    optimizer = torch.optim.Adam(bridge.parameters(), lr=0.01)

    neural_data = torch.randn(num_users, neural_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    # Initial State
    initial_output = bridge(neural_data, edge_index)
    initial_h = initial_output["grid_entropy"].item()
    initial_phi = initial_output["phi_q"].item()

    # Train for a few steps to increase Phi and decrease H
    for _ in range(10):
        optimizer.zero_grad()
        output = bridge(neural_data, edge_index)
        loss = calculate_ngr_loss(output["phi_q"], output["grid_entropy"], target_phi=1.5)
        loss.backward()
        optimizer.step()

    final_output = bridge(neural_data, edge_index)
    final_h = final_output["grid_entropy"].item()
    final_phi = final_output["phi_q"].item()

    # Note: With only 10 steps and random init, we expect improvement or stability
    # In a real scenario, this would drive H down significantly.
    # For a test, we just check that the loss is being calculated and used.
    print(f"Initial: H={initial_h:.3f}, Phi={initial_phi:.3f}")
    print(f"Final:   H={final_h:.3f}, Phi={final_phi:.3f}")

    # Loss should be smaller after optimization
    initial_loss = calculate_ngr_loss(initial_output["phi_q"], initial_output["grid_entropy"], target_phi=1.5).item()
    final_loss = calculate_ngr_loss(final_output["phi_q"], final_output["grid_entropy"], target_phi=1.5).item()
    assert final_loss < initial_loss
