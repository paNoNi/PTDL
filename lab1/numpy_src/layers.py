from typing import Tuple

import numpy as np

from lab1.numpy_src.base import Layer
try:
    from lab1.numpy_src.utils.conv_utils import col2im_cython, im2col_cython
except ImportError:
    print('Ooops...')

class FC(Layer):

    def __init__(self, name, in_channels, out_channels):
        super(FC, self).__init__(name)

        self._params = {'weights': {'value': np.random.randn(in_channels, out_channels)},
                        'bias': {'value': np.random.randn(out_channels)}}
        self._params['weights']['grad'] = np.random.randn(in_channels, out_channels)
        self._params['bias']['grad'] = np.random.randn(out_channels)

    def forward(self, x):
        self.in_shape = x.shape
        self.input = x
        return np.dot(x, self._params['weights']['value']) + self._params['bias']['value']

    def backward(self, grad_out):
        dx = np.dot(grad_out, self._params['weights']['value'].T)
        # print(grad_out.max(), dx.max())

        self._params['weights']['grad'] = np.dot(self.input.T, grad_out)
        self._params['bias']['grad'] = np.mean(grad_out, axis=0)

        return dx.reshape(self.in_shape)

    def zero_grad(self):
        self._params['weights']['grad'].fill(0)
        self._params['bias']['grad'].fill(0)

    def update(self, new_params: dict):
        self._params = new_params


class Conv2D(Layer):
    def __init__(self, name, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Conv2D, self).__init__(name)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding

        self._params = {
            'weights': {'value': np.random.randn(out_channels, in_channels, kernel_size, kernel_size)},
            'bias': {'value': np.random.randn(out_channels)}}
        self._params['weights']['grad'] = np.random.randn(*self._params['weights']['value'].shape)
        self._params['bias']['grad'] = np.random.randn(*self._params['bias']['value'].shape)

    def forward(self, x: np.array) -> np.array:
        """
        :param a_prev - 4D tensor with shape (n, h_in, w_in, c)
        :output 4D tensor with shape (n, h_out, w_out, n_f)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        a_prev = np.transpose(x, axes=(0, 2, 3, 1))
        self._a_prev = np.array(a_prev, copy=True)
        b, c, h, w = x.shape
        h_out, w_out = (h + self.padding * 2 - self.ksize) // self.stride + 1, \
                       (w + self.padding * 2 - self.ksize) // self.stride + 1
        weights = np.transpose(self._params['weights']['value'], axes=(2, 3, 1, 0))
        h_f, w_f, _, n_f = weights.shape
        w = np.transpose(weights, (3, 2, 0, 1))

        self._cols = im2col_cython(
            np.moveaxis(a_prev, -1, 1),
            h_f,
            w_f,
            self.padding,
            self.stride
        )

        result = w.reshape((n_f, -1)).dot(self._cols)
        output = result.reshape(n_f, h_out, w_out, b)
        output = output.transpose(3, 0, 1, 2)
        # for i in range(output.shape[1]):
        #     output[:, i, :, :] = output[:, i, :, :] + self._params['bias']['value'][i]
        return output

    def backward(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 4D tensor with shape (n, h_out, w_out, n_f)
        :output 4D tensor with shape (n, h_in, w_in, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        n = da_curr.shape[0],
        weights = np.transpose(self._params['weights']['value'], axes=(2, 3, 0, 1))
        h_f, w_f, n_f, _ = weights.shape

        self._params['bias']['grad'] = da_curr.sum(axis=(0, 2, 3)) / n
        da_curr_reshaped = da_curr.transpose(3, 1, 2, 0).reshape(n_f, -1)

        w = np.transpose(weights, (3, 2, 0, 1))
        dw = da_curr_reshaped.dot(self._cols.T).reshape(w.shape)
        self._params['weights']['grad'] = np.transpose(dw, (1, 0, 2, 3))

        output_cols = w.reshape(n_f, -1).T.dot(da_curr_reshaped)

        a_prev = np.moveaxis(self._a_prev, -1, 1)
        output = col2im_cython(
            output_cols,
            a_prev.shape[0],
            a_prev.shape[1],
            a_prev.shape[2],
            a_prev.shape[3],
            h_f,
            w_f,
            self.padding,
            self.stride
        )
        return output

    def zero_grad(self):
        self._params['weights']['grad'] = np.zeros_like(self._params['weights']['grad'], dtype='float64')
        self._params['bias']['grad'] = np.zeros_like(self._params['bias']['grad'], dtype='float64')

    def update(self, new_params: dict):
        self._params = new_params

    def im2col(self, x, k_size, stride):
        b, c, h, w = x.shape
        image_col = []
        for n in range(b):
            for i in range(0, h - k_size + 1, stride):
                for j in range(0, w - k_size + 1, stride):
                    col = x[n, :, i:i + k_size, j:j + k_size].reshape(-1)
                    image_col.append(col)

        return np.array(image_col)


class MaxPooling(Layer):
    def __init__(self, name, ksize=2, stride=2):
        super(MaxPooling, self).__init__(name)
        self.ksize = (ksize, ksize)
        self.stride = stride
        self._a = None
        self._cache = {}

    def forward(self, x):
        x_ = np.transpose(x, axes=(0, 2, 3, 1))
        self._a = np.array(x_, copy=True)
        n, h_in, w_in, c = x_.shape
        h_pool, w_pool = self.ksize
        h_out = 1 + (h_in - h_pool) // self.stride
        w_out = 1 + (w_in - w_pool) // self.stride
        output = np.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                a_prev_slice = x_[:, h_start:h_end, w_start:w_end, :]
                self._save_mask(x=a_prev_slice, cords=(i, j))
                output[:, i, j, :] = np.array(np.max(a_prev_slice, axis=(1, 2)))
        return output.transpose(0, 3, 1, 2)
        # n, c, h, w = x.shape
        # out = np.zeros([n, c, h // self.stride, w // self.stride])
        # self.index = np.zeros_like(x)
        # for b in range(n):
        #     for d in range(c):
        #         for i in range(0, h - self.ksize, self.stride):
        #             for j in range(0, w - self.ksize, self.stride):
        #                 out[b, d, i // self.stride, j // self.stride] = np.max(
        #                     x[b, d, i:i + self.ksize, j:j + self.ksize])
        #                 index = np.argmax(x[b, d, i:i + self.ksize, j:j + self.ksize])
        #                 self.index[b, d, i + index // self.ksize, j + index % self.ksize] = 1
        # return out

    def backward(self, grad_out):
        output = np.zeros_like(self._a)
        grad_out_ = np.transpose(grad_out, axes=(0, 2, 3, 1))
        _, h_out, w_out, _ = grad_out_.shape
        h_pool, w_pool = self.ksize

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                output[:, h_start:h_end, w_start:w_end, :] += grad_out_[:, i:i + 1, j:j + 1, :] * self._cache[(i, j)]
        return output.transpose(0, 3, 1, 2)

    def _save_mask(self, x: np.array, cords: Tuple[int, int]) -> None:
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self._cache[cords] = mask


class Flatten(Layer):

    def forward(self, input):
        self.input_shape = input.shape
        return np.reshape(input, (input.shape[0], int(input.size / input.shape[0])))

    def backward(self, grad_out):
        return np.reshape(grad_out, self.input_shape)


class BatchNormLayer(Layer):

    def __init__(self, dims: int, name) -> None:
        super(BatchNormLayer, self).__init__(name)
        self.gamma = np.ones((1, dims), dtype="float64")
        self.bias = np.zeros((1, dims), dtype="float64")
        self.epsilon = 10 ** -3

        self.running_mean_x = np.zeros(0)
        self.running_var_x = np.zeros(0)

        # forward params
        self.var_x = np.zeros(0)
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.num_examples = 0
        self.mean_x = np.zeros(0)
        self.running_avg_gamma = 0.9

        # backward params
        self.gamma_grad = np.zeros(0)
        self.bias_grad = np.zeros(0)

    def update_running_variables(self) -> None:
        is_mean_empty = np.array_equal(np.zeros(0), self.running_mean_x)
        is_var_empty = np.array_equal(np.zeros(0), self.running_var_x)
        if is_mean_empty != is_var_empty:
            raise ValueError("Mean and Var running averages should be "
                             "initilizaded at the same time")
        if is_mean_empty:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
        else:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + (1. - gamma) * self.var_x

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.num_examples = x.shape[0]
        if self.is_train():
            self.mean_x = np.mean(x, axis=0, keepdims=True)
            square = np.power(x - self.mean_x, 2)
            self.var_x = np.mean(square, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            self.mean_x = self.running_mean_x.copy()
            self.var_x = self.running_var_x.copy()

        self.var_x += self.epsilon
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = x - self.mean_x
        self.standard_x = self.x_minus_mean / self.stddev_x
        return self.gamma * self.standard_x + self.bias

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        standard_grad = grad_input * self.gamma

        var_grad = np.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3 / 2),
                          axis=0, keepdims=True)
        stddev_inv = 1 / self.stddev_x
        aux_x_minus_mean = 2 * self.x_minus_mean / self.num_examples

        mean_grad = (np.sum(standard_grad * -stddev_inv, axis=0,
                            keepdims=True) +
                     var_grad * np.sum(-aux_x_minus_mean, axis=0,
                                       keepdims=True))

        self.gamma_grad = np.sum(grad_input * self.standard_x, axis=0,
                                 keepdims=True)
        self.bias_grad = np.sum(grad_input, axis=0, keepdims=True)

        return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + mean_grad / self.num_examples


class Dropout(Layer):
    def __init__(self, name, prob: float = 0.5):
        super(Dropout, self).__init__(name)
        self.prob = prob
        self.__d_mask = None

    def forward(self, x):
        if not self.is_train():
            return x
        d_mask = np.random.random(x.shape)
        d_mask[d_mask >= self.prob] = 1
        d_mask[d_mask < self.prob] = 0
        self.__d_mask = d_mask
        x = x * d_mask
        x /= self.prob
        return x

    def backward(self, grad_out):
        grad_out = grad_out * self.__d_mask
        grad_out /= self.prob
        return grad_out

class BatchNorm2D(Layer):
    def __init__(self, name: str, n_channels, epsilon=1e-5):
        super(BatchNorm2D, self).__init__(name)
        self.epsilon = epsilon
        self.n_channels = n_channels
        self.weight = dict()
        self.weight_update = dict()
        self.cache = dict()
        self.grad = dict()
        self.weight["gamma"] = np.ones(shape=(1, n_channels, 1, 1))
        self.weight["beta"] = np.zeros(shape=(1, n_channels, 1, 1))


    def forward(self, X):
        """
        Forward pass for the 2D batchnorm layer.
        Args:
            X: numpy.ndarray of shape (n_batch, n_channels, height, width).
        Returns_
            Y: numpy.ndarray of shape (n_batch, n_channels, height, width).
                Batch-normalized tensor of X.
        """
        mean = np.mean(X, axis=(2, 3), keepdims=True)
        var = np.var(X, axis=(2, 3), keepdims=True) + self.epsilon
        invvar = 1.0 / var
        sqrt_invvar = np.sqrt(invvar)
        centered = X - mean
        scaled = centered * sqrt_invvar
        normalized = scaled * self.weight["gamma"] + self.weight["beta"]

        # caching intermediate results for backprop
        self.cache["mean"] = mean
        self.cache["var"] = var
        self.cache["invvar"] = invvar
        self.cache["sqrt_invvar"] = sqrt_invvar
        self.cache["centered"] = centered
        self.cache["scaled"] = scaled
        self.cache["normalized"] = normalized
        self.grad = self.local_grad(X)

        return normalized

    def backward(self, dY):
        """
        Backward pass for the 2D batchnorm layer. Calculates global gradients
        for the input and the parameters.
        Args:
            dY: numpy.ndarray of shape (n_batch, n_channels, height, width).
        Returns:
            dX: numpy.ndarray of shape (n_batch, n_channels, height, width).
                Global gradient wrt the input X.
        """
        # global gradients of parameters
        dgamma = np.sum(self.cache["scaled"] * dY, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dY, axis=(0, 2, 3), keepdims=True)

        # caching global gradients of parameters
        self.weight_update["gamma"] = dgamma
        self.weight_update["beta"] = dbeta

        # global gradient of the input
        dX = self.grad["X"] * dY

        return dX

    def local_grad(self, X):
        """
        Calculates the local gradient for X.
        Args:
            dY: numpy.ndarray of shape (n_batch, n_channels, height, width).
        Returns:
            grads: dictionary of gradients.
        """
        # global gradient of the input
        N, C, H, W = X.shape
        # ppc = pixels per channel, useful variable for further computations
        ppc = H * W

        # gradient for 'denominator path'
        dsqrt_invvar = self.cache["centered"]
        dinvvar = (1.0 / (2.0 * np.sqrt(self.cache["invvar"]))) * dsqrt_invvar
        dvar = (-1.0 / self.cache["var"] ** 2) * dinvvar
        ddenominator = (X - self.cache["mean"]) * (2 * (ppc - 1) / ppc ** 2) * dvar

        # gradient for 'numerator path'
        dcentered = self.cache["sqrt_invvar"]
        dnumerator = (1.0 - 1.0 / ppc) * dcentered

        dX = ddenominator + dnumerator
        grads = {"X": dX}
        return grads