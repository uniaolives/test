# src/papercoder_kernel/glp/primordial.py
import numpy as np
from ..core.scratch.tensor import ScratchTensor
from ..core.scratch.layers import Linear, ReLU, Softmax
from ..core.scratch.nn import CrossEntropyLoss, SGD

class PrimordialGLP:
    """
    "Old School" neural network implementation for Linear A.
    No frameworks. Built from scratch with manual backprop.
    """
    def __init__(self, input_dim=16, hidden1=32, hidden2=16, output=50):
        self.l1 = Linear(input_dim, hidden1)
        self.r1 = ReLU()
        self.l2 = Linear(hidden1, hidden2)
        self.r2 = ReLU()
        self.l3 = Linear(hidden2, output)
        self.softmax = Softmax()

        self.params = [
            self.l1.weights, self.l1.bias,
            self.l2.weights, self.l2.bias,
            self.l3.weights, self.l3.bias
        ]

    def forward(self, X_tensor: ScratchTensor):
        h1 = self.r1.forward(self.l1.forward(X_tensor))
        h2 = self.r2.forward(self.l2.forward(h1))
        out = self.softmax.forward(self.l3.forward(h2))
        return out

    def train_epoch(self, X_data, Y_data, optimizer, loss_fn):
        X = ScratchTensor(X_data)
        Y = ScratchTensor(Y_data)

        # Forward
        pred = self.forward(X)

        # Loss
        loss = loss_fn(pred, Y)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Step
        optimizer.step()

        return loss.data
