# src/neural/validate_fusion.py
import torch
from src.neural.bio_graph_network import OmegaFusion
import os

def validate():
    print("--- 🜏 ARKHE(N) NEURAL VALIDATION ---")

    # 1. Test Model Initialization and Export Consistency
    print("[1] Testing Eager vs. Traced consistency...")
    biometric_dim = 4
    model = OmegaFusion(biometric_dim=biometric_dim)
    model.eval()

    # Dummy inputs
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 1)
    batch = torch.zeros(3, dtype=torch.long)

    with torch.no_grad():
        phi_q, temporal_sig, zk_proofs, global_state = model(x, edge_index, edge_attr, batch)

    # Trace it
    traced_model = torch.jit.trace(model, (x, edge_index, edge_attr, batch))

    with torch.no_grad():
        l_phi_q, l_temporal_sig, l_zk_proofs, l_global_state = traced_model(x, edge_index, edge_attr, batch)

    print(f"    Eager φ_q: {phi_q.item():.4f}")
    print(f"    Traced φ_q: {l_phi_q.item():.4f}")

    assert torch.allclose(phi_q, l_phi_q), "Output mismatch between eager and traced model!"
    print("    Consistency Verification: SUCCESS")

    # 2. Verify existence of the production model
    print("[2] Checking production model 'teknet_omega.pt'...")
    if os.path.exists("teknet_omega.pt"):
        print("    Found teknet_omega.pt")
        try:
            prod_model = torch.jit.load("teknet_omega.pt")
            p_phi_q, _, _, _ = prod_model(x, edge_index, edge_attr, batch)
            print(f"    Production φ_q (on random input): {p_phi_q.item():.4f}")
            print("    Production Load Verification: SUCCESS")
        except Exception as e:
            print(f"    Production Load Verification: FAILED - {e}")
            return False
    else:
        print("    teknet_omega.pt NOT FOUND")
        return False

    print("--- 🜏 ALL NEURAL TESTS PASSED ---")
    return True

if __name__ == "__main__":
    if validate():
        exit(0)
    else:
        exit(1)
