import numpy as np


class Value:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class Layer:
    def __init__(self, *args, **kwargs):
        self.name = 'layer'

    def forward(self, x: np.ndarray):
        pass

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        pass

    @property
    def params(self):
        return {}

    def zero_grad(self):
        pass


class Sequential(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = args

    def forward(self, x: np.ndarray):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, x: np.ndarray):
        grad = x
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    @property
    def params(self):
        params_all = list()
        for layer in self.layers:
            params_all.append(layer.params)
        return params_all


class FCLayer(Layer):
    def __init__(self, n_input, n_output, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = Value(0.001 * np.random.randn(n_input, n_output))
        self.bias = Value(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = Value(X)
        xw = X.dot(self.weights.value) + self.bias.value
        return xw

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        self.weights.grad = self.X.value.transpose().dot(d_out)
        self.X.grad = d_out.dot(self.weights.value.transpose())
        self.bias.grad = np.array([np.sum(d_out, axis=0).transpose()])

        return self.X.grad

    @property
    def params(self):
        return {'W': self.weights, 'B': self.bias}

    def zero_grad(self):
        if self.X is not None:
            self.X.grad = np.zeros_like(self.X.value)
        self.weights.grad = np.zeros_like(self.weights.value)
        self.bias.grad = np.zeros_like(self.bias.value)


class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride=1, padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = (kernel_size, kernel_size)

        self.kernels = Value(0.001 * np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Value(0.001 * np.random.randn(out_channels, 1, kernel_size, kernel_size))

    def forward(self, x: np.ndarray):
        features_maps = np.zeros((x.shape[0], self.out_channels,
                                  (x.shape[2] - self.kernel_size[0] + 2 * self.padding) // self.stride + 1,
                                  (x.shape[3] - self.kernel_size[1] + 2 * self.padding) // self.stride + 1))

        return features_maps
