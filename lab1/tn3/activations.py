import numpy as np

from lab1.tn3.base import Layer


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        self.input = input
        return np.maximum(input, 0)

    def backward(self, grad_out):
        grad_out[self.input <= 0] = 0
        return grad_out


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)
        self.output = None

    def forward(self, x):
        self.output = np.piecewise(
            x,
            [x > 0],
            [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
        )
        return self.output

    def backward(self, grad):
        grad = grad * self.output * (1 - self.output)
        return grad


class Tanh(Layer):
    def __init__(self, name):
        super(Tanh, self).__init__(name)

    def forward(self, input):
        a = np.exp(input)
        b = np.exp(-input)
        self.output = (a - b) / (a + b)
        return self.output

    def backward(self, grad):
        grad = grad * (1 - self.output * self.output)
        return grad


class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)
        self.__last_softmax = None

    def forward(self, x: np.ndarray):
        e = np.exp(x.transpose() - np.max(x, axis=1)).transpose()
        self.__last_softmax = e / np.sum(e, axis=1, keepdims=True)
        return self.__last_softmax

    def backward(self, grad_out):
        # z, da shapes - (m, n)
        m, n = self.__last_softmax.shape
        # First we create for each example feature vector, it's outer product with itself
        # ( p1^2  p1*p2  p1*p3 .... )
        # ( p2*p1 p2^2   p2*p3 .... )
        # ( ...                     )
        tensor1 = np.einsum('ij,ik->ijk', self.__last_softmax, self.__last_softmax)  # (m, n, n)
        # Second we need to create an (n,n) identity of the feature vector
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        tensor2 = np.einsum('ij,jk->ijk', self.__last_softmax, np.eye(n, n))  # (m, n, n)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
        dSoftmax = tensor2 - tensor1
        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
        dz = np.einsum('ijk,ik->ij', dSoftmax, grad_out)  # (m, n)
        return dz
