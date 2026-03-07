# src/neural/export_model.py
import torch
from src.neural.bio_graph_network import OmegaFusion

def export_omega_fusion():
    print("Initializing OmegaFusion model...")
    model = OmegaFusion(biometric_dim=4, hidden_dim=128, num_layers=3)
    model.eval()

    # Create dummy inputs for tracing
    # N=1 node, biometric_dim=4
    x = torch.randn(1, 4)
    # 0 edges
    edge_index = torch.empty((2, 0), dtype=torch.long)
    # 0 edge attributes
    edge_attr = torch.empty((0, 1))
    # Batch vector
    batch = torch.zeros(1, dtype=torch.long)

    print("Tracing model with TorchScript...")
    # Using tracing for simplicity, but script might be better for dynamic graphs
    # However, PyG models often require careful handling with TorchScript.
    try:
        traced_model = torch.jit.trace(model, (x, edge_index, edge_attr, batch))
        traced_model.save("teknet_omega.pt")
        print("SUCCESS: Model saved to teknet_omega.pt")
    except Exception as e:
        print(f"ERROR during tracing: {e}")
        print("Attempting torch.jit.script...")
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save("teknet_omega.pt")
            print("SUCCESS: Model saved to teknet_omega.pt via scripting")
        except Exception as e2:
            print(f"ERROR during scripting: {e2}")

if __name__ == "__main__":
    export_omega_fusion()
