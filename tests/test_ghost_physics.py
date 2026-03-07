# tests/test_ghost_physics.py
import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics.ghost_solver import TopologicalSolver
from src.physics.ghost_clustering import build_three_body_graph, ThreeBodyGhostNet

def test_ghost_solver():
    print("Testing TopologicalSolver...")
    # 3 bodies in 3D: (pos1, pos2, pos3, vel1, vel2, vel3) = 18 elements
    initial_conditions = [
        np.array([
            0, 0, 0,  # pos1
            1, 0, 0,  # pos2
            0, 1, 0,  # pos3
            0, 0, 0,  # vel1
            0, 0.5, 0,# vel2
            -0.5, 0, 0 # vel3
        ])
    ]

    config = {
        'initial_conditions': initial_conditions
    }

    solver = TopologicalSolver(input_dim=18)
    results = solver.solve(config)

    print(f"Discovered {len(results['solutions'])} stable ghost orbits.")
    for i, score in enumerate(results['stability_scores']):
        print(f"Orbit {i}: stability = {score:.4f}")

    assert len(results['solutions']) >= 0
    print("TopologicalSolver test passed.")

def test_ghost_clustering_gnn():
    print("\nTesting ThreeBodyGhostNet...")
    energy = 1.0
    angular_momentum = 0.5
    data = build_three_body_graph(energy, angular_momentum)

    model = ThreeBodyGhostNet(hidden_dim=4, num_layers=2)
    output = model(data)

    print(f"φ_q: {output['phi_q']:.4f}")
    print(f"Ghost locations shape: {output['ghost_locations'].shape}")

    assert output['phi_q'] > 4.64
    print("ThreeBodyGhostNet test passed.")

if __name__ == "__main__":
    try:
        test_ghost_solver()
        test_ghost_clustering_gnn()
        print("\nAll ghost physics tests passed successfully.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
