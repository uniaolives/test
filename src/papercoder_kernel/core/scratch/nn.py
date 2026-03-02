# src/papercoder_kernel/core/scratch/nn.py
import numpy as np
from .tensor import ScratchTensor

class MSELoss:
    """
    Manual Mean Squared Error Loss.
    """
    def __call__(self, pred: ScratchTensor, target: ScratchTensor):
        diff = pred.data - target.data
        loss_val = np.mean(diff**2)
        return ScratchTensor(loss_val, requires_grad=pred.requires_grad, creators=[pred, target], op="mse_loss")

class CrossEntropyLoss:
    """
    Manual Cross Entropy Loss.
    """
    def __call__(self, pred: ScratchTensor, target: ScratchTensor):
        loss_val = -np.sum(target.data * np.log(pred.data + 1e-10)) / target.data.shape[0]
        return ScratchTensor(loss_val, requires_grad=pred.requires_grad, creators=[pred, target], op="ce_loss")

class SGD:
    """
    Manual Stochastic Gradient Descent.
    """
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)
