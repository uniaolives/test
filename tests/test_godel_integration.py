import torch
import torch_geometric
from src.godel.ghost_clustering import GhostClusterSolver
from src.physics.godel_ghost_solver import GodelGhostNet, build_three_body_graph
from src.physics.godel_flow import GodelFlow

def test_ghost_cluster_solver():
    print("Testing GhostClusterSolver with N=10 nodes...")
    solver = GhostClusterSolver(dim=1024)
    # 3 bodies * 6 states (pos, vel) = 18
    initial_states = torch.randn(10, 18)
    output = solver(initial_states)
    print(f"Ghost probability shape: {output['ghost_probability'].shape}")
    print(f"Phi_q shape: {output['phi_q'].shape}")
    print("GhostClusterSolver test passed.")

def test_godel_ghost_net():
    print("Testing GodelGhostNet...")
    model = GodelGhostNet(state_dim=18, hidden_dim=1024)
    data = build_three_body_graph(num_nodes=10)
    output = model(data)
    print(f"Ghost score shape: {output['ghost_score'].shape}")
    print(f"Stable state shape: {output['stable_state'].shape}")
    print("GodelGhostNet test passed.")

def test_godel_flow():
    print("Testing GodelFlow...")
    flow = GodelFlow(dim=1024)
    x = torch.randn(1, 1024)
    x_new = flow(x)
    print(f"Flow update: {torch.norm(x - x_new)}")
    print("GodelFlow test passed.")

if __name__ == "__main__":
    try:
        test_ghost_cluster_solver()
        test_godel_ghost_net()
        test_godel_flow()
        print("\nAll Gödelian Python components verified successfully.")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
