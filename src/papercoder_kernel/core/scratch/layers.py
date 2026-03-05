# src/papercoder_kernel/core/scratch/layers.py
import numpy as np
from .tensor import ScratchTensor

class Layer:
    def forward(self, input):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError

class Linear(Layer):
    """
    Manual Linear layer.
    y = xW + b
    """
    def __init__(self, in_features, out_features):
        # He initialization for ReLU
        self.weights = ScratchTensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
        self.bias = ScratchTensor(np.zeros((1, out_features)), requires_grad=True)
        self.input = None

    def forward(self, input):
        self.input = input
        # input: [batch, in], weights: [in, out] -> [batch, out]
        return input.matmul(self.weights) + self.bias

class ReLU(Layer):
    """
    Manual ReLU layer.
    """
    def forward(self, input):
        self.input = input
        data = np.maximum(0, input.data)
        return ScratchTensor(data, requires_grad=input.requires_grad, creators=[input], op="relu")

    # Custom backward for ReLU since it's non-linear
    # Note: ScratchTensor.backward needs to be updated to handle non-linear ops
    # For prototype, we'll implement backward logic inside the tensor or here.

class Sigmoid(Layer):
    def forward(self, input):
        self.input = input
        data = 1 / (1 + np.exp(-input.data))
        return ScratchTensor(data, requires_grad=input.requires_grad, creators=[input], op="sigmoid")

class Softmax(Layer):
    """
    Manual Softmax.
    """
    def forward(self, input):
        self.input = input
        exps = np.exp(input.data - np.max(input.data, axis=1, keepdims=True))
        data = exps / np.sum(exps, axis=1, keepdims=True)
        return ScratchTensor(data, requires_grad=input.requires_grad, creators=[input], op="softmax")

class Conv2D(Layer):
    """
    Manual 2D Convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # He initialization
        limit = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        weights_data = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit
        self.weights = ScratchTensor(weights_data, requires_grad=True)
        self.bias = ScratchTensor(np.zeros((out_channels, 1)), requires_grad=True)

    def forward(self, input):
        return input.conv2d(self.weights, self.bias, self.stride, self.padding)

class MaxPool2D(Layer):
    """
    Manual 2D Max Pooling layer.
    """
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, input):
        return input.maxpool2d(self.kernel_size, self.stride)

class Flatten(Layer):
    """
    Flatten layer to transition from Conv to Linear.
    """
    def forward(self, input):
        return input.flatten()
