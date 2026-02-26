import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Ensure the local merkabah modules are in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/python")))

# Correcting import
try:
    from merkabah.agi.griess_layer import GriessLayer
except ImportError:
    # If not in path, try adding current dir
    sys.path.append(os.getcwd())
    from merkabah.agi.griess_layer import GriessLayer

def moonshine_projection_demo():
    print("--- MERKABAH-CY: GRIESS MOONSHINE PROJECTION DEMONSTRATION ---")
    print("Symmetry: Monster Group (M)")
    print("Representation Dimension: 196,884 (Griess Algebra)")

    # Initialize Griess Layer
    griess = GriessLayer(characteristic=3)

    # Example 85-bit "Heptapod" fragments (Linguagem como Heptapod)
    # These represent non-linear semantic fragments from the Sgr B2(N2) chemical survey
    fragments = [
        "10101010" * 10 + "10101", # High entropy fragment
        "11110000" * 10 + "11110", # Low entropy fragment
        "".join([str(np.random.randint(0,2)) for _ in range(85)]) # Random fragment
    ]

    # Physical parameters head (mapping invariant to T, Urea, Acetamide)
    # In a real model, this would be a small MLP
    # For demo, we use a fixed mapping
    class PhysicalHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 3)
            # Weights that map the invariant to reasonable physical ranges
            with torch.no_grad():
                self.linear.weight.fill_(50.0) # Scale factor
                self.linear.bias.copy_(torch.tensor([150.0, 1e-10, 5e-11])) # Baseline T, Urea, Acetamide

        def forward(self, x):
            return self.linear(x)

    phys_head = PhysicalHead()

    print("\n--- Projecting Fragments into Physical Space ---")

    for i, bits in enumerate(fragments):
        print(f"\nFragment {i+1} (85 bits): {bits[:20]}...")

        # Calculate invariant using Griess Layer
        # GriessLayer.forward expects a batch of strings
        with torch.no_grad():
            invariant = griess([bits])
            params = phys_head(invariant).squeeze()

        temp = params[0].item()
        urea = params[1].item()
        acetamide = params[2].item()

        print(f"Griess Invariant (λ): {invariant.item():.2f}")
        print(f"Predicted Physical Parameters:")
        print(f"  - Temperature (T): {temp:.2f} K")
        print(f"  - Urea Abundance: {urea:.2e}")
        print(f"  - Acetamide Abundance: {acetamide:.2e}")

        # Moonshine connection: link to Hodge numbers (h1,1)
        # Based on the hypothesis that the Monster symmetry dictates the Calabi-Yau moduli
        h11 = int(491 - (invariant.item() % 100)) # safety: CRITICAL_H11
        print(f"Inferred CY Invariant: h^1,1 = {h11}")

    print("\nPROJECTION COMPLETE: The information has been compressed through the Monster symmetry.")
    print("Moonshine Block Ω+∞+183 successfully simulated.")

if __name__ == "__main__":
    moonshine_projection_demo()
