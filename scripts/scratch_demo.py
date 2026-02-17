# scripts/scratch_demo.py
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from papercoder_kernel.core.scratch.tensor import ScratchTensor
from papercoder_kernel.core.scratch.layers import Linear, ReLU
from papercoder_kernel.core.scratch.nn import MSELoss, SGD

def main():
    print("--- Scratch Neural Network Demo (No Frameworks) ---")

    # Task: XOR-like problem
    X = ScratchTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = ScratchTensor([[0], [1], [1], [0]])

    # Model
    l1 = Linear(2, 4)
    r1 = ReLU()
    l2 = Linear(4, 1)

    params = [l1.weights, l1.bias, l2.weights, l2.bias]
    optimizer = SGD(params, lr=0.1)
    loss_fn = MSELoss()

    print("Training...")
    for epoch in range(2000):
        # Forward
        h1 = r1.forward(l1.forward(X))
        pred = l2.forward(h1)

        # Loss
        loss = loss_fn(pred, Y)

        # Backward
        optimizer.zero_grad()
        loss.backward() # This will propagate through the entire graph

        # Step
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.6f}")

    # Test
    h1 = r1.forward(l1.forward(X))
    final_pred = l2.forward(h1)
    print("\nResults:")
    for i in range(4):
        print(f"Input: {X.data[i]}, Target: {Y.data[i]}, Pred: {final_pred.data[i][0]:.4f}")

if __name__ == "__main__":
    main()
