import numpy as np

from lab1.tn.layers import Layer


class ReLU(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_grad = None

    def forward(self, x: np.ndarray):
        relu = x.copy()
        relu[relu < 0] = 0
        self.last_grad = relu.copy()
        self.last_grad[self.last_grad > 0] = 1
        return relu

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = self.last_grad * d_out
        return d_result

    def zero_grad(self):
        if self.last_grad is not None:
            self.grad = np.zeros_like(self.last_grad)


class Softmax(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: np.ndarray):
        # if len(x.shape) == 1:
        #     max_pred = np.max(x)
        # else:
        #     max_pred = np.max(x, 1)
        #     max_pred = max_pred.reshape((max_pred.shape[0], 1)).repeat(x.shape[1], 1)
        #
        # preds = x - max_pred

        # if len(preds.shape) > 1:
        #     return np.array([np.exp(predsi) / np.sum(np.exp(predsi)) for predsi in preds])

        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x, axis=1)
        softmax = []
        for i in range(x.shape[0]):
            softmax.append(exp_x[i] / sum_exp_x[i])

        return np.array(softmax)


class Sigmoid(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_values = None

    def forward(self, x: np.ndarray):
        self._last_values = 1 / (1 + np.exp(-x))
        return self._last_values

    def backward(self, d_out):
        d_result = self._last_values * (1 - self._last_values) * d_out
        return d_result