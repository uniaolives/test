import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Ensure the local merkabah modules are in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/python")))

from merkabah.agi.recursive_expansion import RecursiveRankTensor, AutoreferentialLoss

def simulate_projection():
    print("--- MERKABAH-CY: AGI PROJECTION DEMONSTRATION ---")
    print("Targeting Critical Point: h^1,1 = 491") # safety: CRITICAL_H11

    # Initialize Recursive Tensor at the critical dimension
    # (Using 491 as initial dimension to represent the stability limit)
    initial_dim = 491 # safety: CRITICAL_H11
    # Lower entropy threshold to guarantee a leap for the demo
    agi_tensor = RecursiveRankTensor(initial_dim=initial_dim, entropy_threshold=0.1)
    criterion = AutoreferentialLoss()

    # Input with high entropy (complex Hilbert state)
    # Simulating a state that triggers a dimensional leap
    x = torch.randn(1, initial_dim) * 2.0

    print(f"Initial State Rank: {agi_tensor.current_dim}")

    # Forward pass - should trigger expansion
    output = agi_tensor(x)

    print(f"Post-Projection Rank: {agi_tensor.current_dim}")

    if agi_tensor.current_dim > initial_dim:
        print("DIMENSIONAL LEAP SUCCESSFUL: The system has projected into a higher-rank Hilbert space.")
        print(f"New Topology: T^{agi_tensor.current_dim}")
    else:
        print("STABILITY MAINTAINED: No projection required.")

    # Calculate autoreferential loss
    target = torch.zeros_like(output)
    loss = criterion(output, target, agi_tensor.current_dim)
    print(f"Autoreferential Loss: {loss.item():.4f}")

if __name__ == "__main__":
    simulate_projection()
