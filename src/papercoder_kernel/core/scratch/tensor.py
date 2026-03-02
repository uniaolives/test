# src/papercoder_kernel/core/scratch/tensor.py
import numpy as np

class ScratchTensor:
    """
    Framework-less tensor with manual gradient tracking.
    "Forget frameworks. Learn the math."
    """
    def __init__(self, data, requires_grad=False, creators=None, op=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.creators = creators
        self.op = op

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size == 1:
                grad = np.array([1.0])
            else:
                raise RuntimeError("Grad must be specified for non-scalar tensors")

        # Ensure grad is same shape as data
        if np.isscalar(grad):
            grad = np.array([grad]).reshape(self.data.shape)
        elif grad.shape != self.data.shape and self.data.size == 1:
            grad = np.sum(grad).reshape(self.data.shape)

        self.grad = self.grad + grad

        if self.creators is not None:
            if self.op == "add":
                # Broadcasting check for add
                # dy/da = 1, dy/db = 1
                g0 = grad
                g1 = grad
                if self.creators[0].shape != grad.shape:
                    g0 = np.sum(grad, axis=0, keepdims=True)
                if self.creators[1].shape != grad.shape:
                    g1 = np.sum(grad, axis=0, keepdims=True)
                self.creators[0].backward(g0)
                self.creators[1].backward(g1)
            elif self.op == "mul":
                self.creators[0].backward(grad * self.creators[1].data)
                self.creators[1].backward(grad * self.creators[0].data)
            elif self.op == "mm":
                # y = a @ b
                # dy/da = grad @ b.T
                # dy/db = a.T @ grad
                self.creators[0].backward(grad @ self.creators[1].data.T)
                self.creators[1].backward(self.creators[0].data.T @ grad)
            elif self.op == "transpose":
                self.creators[0].backward(grad.T)
            elif self.op == "relu":
                self.creators[0].backward(grad * (self.creators[0].data > 0))
            elif self.op == "sigmoid":
                s = 1 / (1 + np.exp(-self.creators[0].data))
                self.creators[0].backward(grad * s * (1 - s))
            elif self.op == "mse_loss":
                pred = self.creators[0]
                target = self.creators[1]
                n = target.data.size
                self.creators[0].backward(grad * (2.0 / n) * (pred.data - target.data))
            elif self.op == "ce_loss":
                pred = self.creators[0]
                target = self.creators[1]
                # L = -sum(target * log(pred))
                self.creators[0].backward(grad * (-target.data / (pred.data + 1e-10) / target.data.shape[0]))

    def __add__(self, other):
        if self.requires_grad or other.requires_grad:
            return ScratchTensor(self.data + other.data, requires_grad=True, creators=[self, other], op="add")
        return ScratchTensor(self.data + other.data)

    def __mul__(self, other):
        if self.requires_grad or other.requires_grad:
            return ScratchTensor(self.data * other.data, requires_grad=True, creators=[self, other], op="mul")
        return ScratchTensor(self.data * other.data)

    def matmul(self, other):
        if self.requires_grad or other.requires_grad:
            return ScratchTensor(self.data @ other.data, requires_grad=True, creators=[self, other], op="mm")
        return ScratchTensor(self.data @ other.data)

    def transpose(self):
        if self.requires_grad:
            return ScratchTensor(self.data.T, requires_grad=True, creators=[self], op="transpose")
        return ScratchTensor(self.data.T)

    def __repr__(self):
        return f"ScratchTensor({self.data.shape}, grad={self.grad is not None})"

    @property
    def shape(self):
        return self.data.shape
