# examples/xor_scratch.py
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from papercoder_kernel.core.scratch.tensor import ScratchTensor
from papercoder_kernel.core.scratch.layers import Linear, ReLU, Sigmoid
from papercoder_kernel.core.scratch.nn import MSELoss, SGD

def main():
    print("--- Solving XOR from Scratch ---")

    # XOR dataset
    X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_data = np.array([[0], [1], [1], [0]])

    # Model: 2 -> 4 -> 1
    l1 = Linear(2, 4)
    r1 = ReLU()
    l2 = Linear(4, 1)
    s1 = Sigmoid()

    loss_fn = MSELoss()
    optimizer = SGD([l1.weights, l1.bias, l2.weights, l2.bias], lr=0.5)

    epochs = 2000
    for epoch in range(epochs):
        X = ScratchTensor(X_data)
        Y_true = ScratchTensor(Y_data)

        # Forward
        z1 = l1.forward(X)
        a1 = r1.forward(z1)
        z2 = l2.forward(a1)
        pred = s1.forward(z2)

        loss = loss_fn(pred, Y_true)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

    print("\n--- Final Predictions ---")
    final_pred = s1.forward(l2.forward(r1.forward(l1.forward(ScratchTensor(X_data)))))
    for i in range(4):
        print(f"Input: {X_data[i]}, Target: {Y_data[i][0]}, Prediction: {final_pred.data[i][0]:.4f}")

if __name__ == "__main__":
    main()
