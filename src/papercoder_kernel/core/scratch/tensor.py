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
                    g0 = np.sum(grad, axis=0, keepdims=True).reshape(self.creators[0].shape)
                if self.creators[1].shape != grad.shape:
                    g1 = np.sum(grad, axis=0, keepdims=True).reshape(self.creators[1].shape)
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
            elif self.op == "softmax":
                Y = self.data
                dX = np.zeros_like(Y)
                for n in range(Y.shape[0]):
                    y = Y[n].reshape(-1, 1)
                    jacobian = np.diagflat(y) - np.dot(y, y.T)
                    dX[n] = np.dot(jacobian, grad[n].reshape(-1, 1)).flatten()
                self.creators[0].backward(dX)
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
            elif self.op == "conv2d":
                X, W_t, b = self.creators
                stride, padding = self.stride, self.padding
                N, C, H_in, W_in = X.shape
                F, _, HH, WW = W_t.shape
                _, _, H_out, W_out = self.shape
                # dBias: [F, 1]
                if b.requires_grad:
                    db = np.sum(grad, axis=(0, 2, 3)).reshape(b.shape)
                    b.backward(db)
                # dWeights: [F, C, HH, WW]
                if W_t.requires_grad:
                    dW = np.zeros_like(W_t.data)
                    x_padded = np.pad(X.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
                    for i in range(H_out):
                        for j in range(W_out):
                            h_start, w_start = i * stride, j * stride
                            x_slice = x_padded[:, :, h_start:h_start+HH, w_start:w_start+WW]
                            for f in range(F):
                                dW[f] += np.sum(x_slice * grad[:, f, i, j][:, None, None, None], axis=0)
                    W_t.backward(dW)
                # dX: [N, C, H, W]
                if X.requires_grad:
                    dX_padded = np.zeros((N, C, H_in + 2*padding, W_in + 2*padding))
                    for i in range(H_out):
                        for j in range(W_out):
                            h_start, w_start = i * stride, j * stride
                            for f in range(F):
                                dX_padded[:, :, h_start:h_start+HH, w_start:w_start+WW] += W_t.data[f] * grad[:, f, i, j][:, None, None, None]
                    X.backward(dX_padded[:, :, padding:padding+H_in, padding:padding+W_in] if padding > 0 else dX_padded)
            elif self.op == "maxpool2d":
                X = self.creators[0]
                arg_max, kernel_size, stride = self.arg_max, self.kernel_size, self.stride
                N, C, H, W = X.shape
                _, _, H_out, W_out = self.shape
                dX = np.zeros_like(X.data)
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * stride, j * stride
                        for n in range(N):
                            for c in range(C):
                                idx = arg_max[n, c, i, j]
                                hh, ww = divmod(idx, kernel_size)
                                dX[n, c, h_start + hh, w_start + ww] += grad[n, c, i, j]
                X.backward(dX)
            elif self.op == "flatten":
                self.creators[0].backward(grad.reshape(self.original_shape))

    def __add__(self, other):
        if not isinstance(other, ScratchTensor): other = ScratchTensor(other)
        if self.requires_grad or other.requires_grad:
            return ScratchTensor(self.data + other.data, requires_grad=True, creators=[self, other], op="add")
        return ScratchTensor(self.data + other.data)

    def __mul__(self, other):
        if not isinstance(other, ScratchTensor): other = ScratchTensor(other)
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

    def conv2d(self, weights, bias, stride=1, padding=0):
        N, C, H_in, W_in = self.shape
        F, _, HH, WW = weights.shape
        x_padded = np.pad(self.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant') if padding > 0 else self.data
        H_out = (H_in + 2 * padding - HH) // stride + 1
        W_out = (W_in + 2 * padding - WW) // stride + 1
        out = np.zeros((N, F, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * stride, j * stride
                x_slice = x_padded[:, :, h_start:h_start+HH, w_start:w_start+WW]
                for f in range(F):
                    out[:, f, i, j] = np.sum(x_slice * weights.data[f], axis=(1, 2, 3)) + bias.data[f]
        res = ScratchTensor(out, requires_grad=self.requires_grad or weights.requires_grad or bias.requires_grad,
                            creators=[self, weights, bias], op="conv2d")
        res.stride, res.padding = stride, padding
        return res

    def maxpool2d(self, kernel_size, stride):
        N, C, H, W = self.shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        out = np.zeros((N, C, H_out, W_out))
        arg_max = np.zeros((N, C, H_out, W_out), dtype=int)
        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * stride, j * stride
                x_slice = self.data[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                reshaped = x_slice.reshape(N, C, -1)
                out[:, :, i, j] = np.max(reshaped, axis=2)
                arg_max[:, :, i, j] = np.argmax(reshaped, axis=2)
        res = ScratchTensor(out, requires_grad=self.requires_grad, creators=[self], op="maxpool2d")
        res.arg_max, res.kernel_size, res.stride = arg_max, kernel_size, stride
        return res

    def flatten(self):
        res = ScratchTensor(self.data.reshape(self.data.shape[0], -1), requires_grad=self.requires_grad, creators=[self], op="flatten")
        res.original_shape = self.data.shape
        return res

    def __repr__(self):
        return f"ScratchTensor({self.data.shape}, grad={self.grad is not None})"

    @property
    def shape(self):
        return self.data.shape
