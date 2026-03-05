# tests/test_scratch_nn.py
import pytest
import numpy as np
from papercoder_kernel.core.scratch.tensor import ScratchTensor
from papercoder_kernel.core.scratch.layers import Linear, ReLU, Sigmoid
from papercoder_kernel.core.scratch.nn import MSELoss

def test_tensor_addition():
    a = ScratchTensor([1.0, 2.0], requires_grad=True)
    b = ScratchTensor([3.0, 4.0], requires_grad=True)
    c = a + b
    assert np.allclose(c.data, [4.0, 6.0])

    c.backward(np.array([1.0, 1.0]))
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0])

def test_tensor_matmul():
    # X: [2, 3], W: [3, 2] -> Y: [2, 2]
    X = ScratchTensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    W = ScratchTensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    Y = X.matmul(W)

    expected_Y = np.array([[1, 2, 3], [4, 5, 6]]) @ np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    assert np.allclose(Y.data, expected_Y)

    grad_out = np.ones((2, 2))
    Y.backward(grad_out)

    # dY/dW = X.T @ grad_out
    expected_grad_W = X.data.T @ grad_out
    assert np.allclose(W.grad, expected_grad_W)

def test_relu_backward():
    x = ScratchTensor([-1.0, 2.0], requires_grad=True)
    r = ReLU()
    y = r.forward(x)
    y.backward(np.array([1.0, 1.0]))

    assert np.allclose(x.grad, [0.0, 1.0])

def test_linear_layer():
    l = Linear(3, 2)
    x = ScratchTensor([[1, 2, 3]], requires_grad=True)
    y = l.forward(x)
    assert y.shape == (1, 2)

    y.backward(np.ones((1, 2)))
    assert l.weights.grad is not None
    assert l.bias.grad is not None

def test_mse_loss():
    loss_fn = MSELoss()
    pred = ScratchTensor([[0.5, 0.5]], requires_grad=True)
    target = ScratchTensor([[1.0, 0.0]])

    loss = loss_fn(pred, target)
    # loss = ((0.5-1)^2 + (0.5-0)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
    assert loss.data == 0.25

    loss.backward()
    # dL/dpred = (2/n) * (pred - target)
    # dL/dpred = (2/2) * ([0.5, 0.5] - [1.0, 0.0]) = [-0.5, 0.5]
    assert np.allclose(pred.grad, [-0.5, 0.5])

def test_numerical_gradient():
    """Verify manual gradient against finite difference."""
    def f(x_data):
        x = ScratchTensor(x_data, requires_grad=True)
        # Some complex op: x^2 + 2x
        y = x * x + x + x
        loss = y.data.sum()
        y.backward(np.ones_like(y.data))
        return loss, x.grad

    x_val = np.array([2.0])
    loss, grad = f(x_val)

    eps = 1e-6
    loss_plus, _ = f(x_val + eps)
    loss_minus, _ = f(x_val - eps)
    num_grad = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(grad, num_grad, atol=1e-5)

def test_conv2d_forward_shape():
    from papercoder_kernel.core.scratch.layers import Conv2D
    # [N, C, H, W] = [1, 1, 5, 5]
    X = ScratchTensor(np.ones((1, 1, 5, 5)), requires_grad=True)
    # [F, C, HH, WW] = [2, 1, 3, 3]
    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
    Y = conv.forward(X)
    # H_out = (5 - 3) / 1 + 1 = 3
    assert Y.shape == (1, 2, 3, 3)

def test_maxpool2d_forward_shape():
    from papercoder_kernel.core.scratch.layers import MaxPool2D
    X = ScratchTensor(np.ones((1, 1, 4, 4)), requires_grad=True)
    pool = MaxPool2D(kernel_size=2, stride=2)
    Y = pool.forward(X)
    assert Y.shape == (1, 1, 2, 2)

def test_conv2d_numerical_gradient():
    from papercoder_kernel.core.scratch.layers import Conv2D
    X_data = np.random.randn(1, 1, 4, 4)

    def f(x_data):
        X = ScratchTensor(x_data, requires_grad=True)
        # Fix weights for deterministic gradient
        W_data = np.ones((1, 1, 2, 2))
        b_data = np.zeros((1, 1))

        W = ScratchTensor(W_data, requires_grad=True)
        b = ScratchTensor(b_data, requires_grad=True)

        Y = X.conv2d(W, b, stride=1, padding=0)
        loss = np.sum(Y.data)
        Y.backward(np.ones_like(Y.data))
        return loss, X.grad

    loss, grad = f(X_data)

    eps = 1e-5
    # Check one element of the gradient
    X_plus = X_data.copy()
    X_plus[0, 0, 0, 0] += eps
    loss_plus, _ = f(X_plus)

    X_minus = X_data.copy()
    X_minus[0, 0, 0, 0] -= eps
    loss_minus, _ = f(X_minus)

    num_grad = (loss_plus - loss_minus) / (2 * eps)
    assert np.allclose(grad[0, 0, 0, 0], num_grad, atol=1e-5)

def test_flatten():
    from papercoder_kernel.core.scratch.layers import Flatten
    X = ScratchTensor(np.ones((2, 3, 4, 4)), requires_grad=True)
    flatten = Flatten()
    Y = flatten.forward(X)
    assert Y.shape == (2, 3 * 4 * 4)

    Y.backward(np.ones_like(Y.data))
    assert X.grad.shape == (2, 3, 4, 4)
    assert np.all(X.grad == 1.0)
