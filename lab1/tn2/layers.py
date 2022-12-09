import math
import numpy as np


def convolve(im, weights):
    # get the dimensions of image and kernel
    kernel_x, kernel_y, in_colors, out_colors = weights.shape
    batch_size, dim_x, dim_y, colors = im.shape

    # allocate an output array
    # the batch_size stays the same, the number of colors is specified in the shape of the filter
    # but the x and y dimensions of the output need to be calculated
    # the formula is:
    # out_x = in_x - filter_x +1
    out = np.empty((batch_size, dim_x - kernel_x + 1, dim_y - kernel_y + 1, out_colors))

    # look at every coordinate in the output
    for i in range(out.shape[1]):
        for j in range(out.shape[2]):
            # at this location, slice a rectangle out of the input image
            # the batch_size and the colors are retained
            # crop has the shape: batch_size, kernel_x, kernel_y, in_colors
            crop = im[:, i:i + kernel_x, j:j + kernel_y]

            # the values in crop will be multiplied by the weights
            # look how the shapes match:
            # crop:   batch_size, x, y, in_colors
            # weights:            x, y, in_colors, out_colors

            # numpy can broadcast this, but ONLY if an extra dimension is added to crop
            # crop now has the shape: batch_size, x, y, in_colors, 1
            crop = np.expand_dims(crop, axis=-1)

            # numpy broadcast magic
            # in parallel along the batch_size
            # matches the x, y and in_colors dimensions and multiplies them pairwise
            res = crop * weights

            # res has the shape: batch_size, x, y, in_colors, out_colors
            # we want to sum along x, y, and in_colors
            # those are the dimensions 1, 2, 3
            # we want to keep the batch_size and the out_colors
            res = np.apply_over_axes(np.sum, res, [1, 2, 3]).reshape(batch_size, -1)

            out[:, i, j] = res

    return out

class ConvLayer:

    def __init__(self, num_filters, filter_width, stride=1, padding=0):
        self.fw = filter_width;
        self.n_f = num_filters;
        self.s = stride;
        self.p = padding
        self.W = None;
        self.b = None

    def forward_propagate(self, input):
        self.input = np.array(input)
        if self.p > 0:  # pad input
            shape = ((0, 0), (self.p, self.p), (self.p, self.p), (0, 0))
            self.input = np.pad(input, shape, mode='constant', constant_values=(0, 0))
        self.W = np.random.random((self.fw, self.fw, self.input.shape[3], self.n_f)) * 0.01
        self.b = np.random.random((1, 1, 1, self.n_f)) * 0.01
        self.n_m = self.input.shape[0]  # number of inputs
        self.ih = self.input.shape[1];
        self.iw = self.input.shape[2]  # input height and width
        self.oh = math.floor((self.ih - self.fw + 2 * self.p) / self.s) + 1  # output height
        self.ow = math.floor((self.iw - self.fw + 2 * self.p) / self.s) + 1  # output width
        self.Z = np.zeros((self.n_m, self.oh, self.ow, self.n_f))
        for i in range(self.n_m):  # iterate over inputs
            for h in range(self.oh):  # iterate over output height
                ih1 = h * self.s;
                ih2 = ih1 + self.fw  # calculate input window height coordinates
                for w in range(self.ow):  # iterate over output width
                    iw1 = w * self.s;
                    iw2 = iw1 + self.fw  # calculate input window width coordinates
                    for f in range(self.n_f):  # iterate over filters
                        self.Z[i, h, w, f] = np.sum(self.input[i, ih1:ih2, iw1:iw2, :] * self.W[:, :, :, f])
                        self.Z += self.b[:, :, :, f]  # calculate output
        return self.Z

    def backpropagate(self, dZ, learning_rate):
        dA_prev = np.zeros((self.n_m, self.ih, self.iw, self.n_f))
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        print(1)
        for i in range(self.n_m):  # iterate over inputs
            for h in range(self.oh):  # iterate over output width
                ih1 = h * self.s
                ih2 = ih1 + self.fw  # calculate input window height coordinates
                for w in range(self.ow):  # iterate over output width
                    iw1 = w * self.s
                    iw2 = iw1 + self.fw  # calculate input window width coordinates
                    for f in range(self.n_f):  # iterate over filters
                        print(dA_prev[i, ih1:ih2, iw1:iw2, :].shape, self.W[:, :, :, f].shape, dZ[i, h, w, f].shape)
                        dA_prev[i, ih1:ih2, iw1:iw2, :] += self.W[:, :, :, f] * dZ[i, h, w, f]
                        dW[:, :, :, f] += self.input[i, ih1:ih2, iw1:iw2, :] * dZ[i, h, w, f]
                        db[:, :, :, f] += dZ[i, h, w, f]
        self.W -= dW * learning_rate
        self.b -= db * learning_rate
        if self.p > 0:  # remove padding
            dA_prev = dA_prev[:, self.p:-self.p, self.p:-self.p, :]
        return dA_prev


class PoolLayer:

    def __init__(self, filter_width, stride=1):
        self.fw = filter_width;
        self.s = stride

    def forward_propagate(self, input):
        im, ih, iw, id = input.shape;
        fw = self.fw;
        s = self.s
        self.n_rows = math.ceil(min(fw, ih - fw + 1) / s)
        self.n_cols = math.ceil(min(fw, iw - fw + 1) / s)
        z_h = int(((ih - fw) / s) + 1);
        z_w = int(((iw - fw) / s) + 1)
        self.Z = np.empty((im, z_h, z_w, id));
        self.input = input
        for t in range(self.n_rows):
            b = ih - (ih - t) % fw
            Z_cols = np.empty((im, int((b - t) / fw), z_w, id))
            for i in range(self.n_cols):
                l = i * s;
                r = iw - (iw - l) % fw
                block = input[:, t:b, l:r, :]
                block = np.array(np.split(block, (r - l) / fw, 2))
                block = np.array(np.split(block, (b - t) / fw, 2))
                block = self.pool(block, 4)
                block = self.pool(block, 3)
                block = np.moveaxis(block, 0, 2)
                block = np.moveaxis(block, 0, 2)
                Z_cols[:, :, i::self.n_cols, :] = block
            self.Z[:, t * s::self.n_rows, :, :] = Z_cols
        self.A = np.abs(self.Z)
        return self.A

    def assemble_block(self, block, t, b, l, r):
        ih = self.input.shape[1];
        iw = self.input.shape[2]
        block = np.repeat(block, self.fw ** 2, 2)
        block = np.array(np.split(block, block.shape[2] / self.fw, 2))
        block = np.moveaxis(block, 0, 2)
        block = np.array(np.split(block, block.shape[2] / self.fw, 2))
        block = np.moveaxis(block, 0, 3)
        return np.reshape(block, (self.input.shape[0], ih - t - b, iw - l - r, self.input.shape[3]))


class PoolLayer_Max(PoolLayer):

    def __init__(self, filter_width, stride=1):
        self.pool = np.max
        super().__init__(filter_width, stride)

    def backpropagate(self, dZ, learning_rate):
        im, ih, iw, id = self.input.shape
        fw = self.fw;
        s = self.s;
        n_rows = self.n_rows;
        n_cols = self.n_cols
        dA_prev = np.zeros(self.input.shape)

        for t in range(n_rows):
            mask_row = self.Z[:, t::n_rows, :, :]
            row = dZ[:, t::self.n_rows, :, :]
            for l in range(self.n_cols):
                b = (ih - t * s) % fw;
                r = (iw - l * s) % fw
                mask = mask_row[:, :, l * s::n_cols, :]
                mask = self.assemble_block(mask, t, b, l, r)
                block = row[:, :, l * s::n_cols, :]
                block = self.assemble_block(block, t, b, l, r)
                mask = (self.input[:, t:ih - b, l:iw - r, :] == mask)
                dA_prev[:, t:ih - b, l:iw - r, :] += block * mask
        return dA_prev


class PoolLayer_Avg(PoolLayer):

    def __init__(self, filter_width, stride=1):
        self.pool = np.mean
        super().__init__(filter_width, stride)

    def backpropagate(self, dZ, learning_rate):
        im, ih, iw, id = self.input.shape
        fw = self.fw;
        s = self.s;
        n_rows = self.n_rows;
        n_cols = self.n_cols
        dA_prev = np.zeros(self.input.shape)

        for t in range(n_rows):
            row = dZ[:, t::n_rows, :, :]
            for l in range(n_cols):
                b = (ih - t * s) % fw;
                r = (iw - l * s) % fw
                block = row[:, :, l * s::n_cols, :]
                block = self.assemble_block(block, t, b, l, r)
                dA_prev[:, t:ih - b, l:iw - r, :] += block / (fw ** 2)
        return dA_prev


class FlatLayer:

    def forward_propagate(self, input):
        self.input_shape = input.shape
        return np.reshape(input, (input.shape[0], int(input.size / input.shape[0])))

    def backpropagate(self, dZ, learning_rate):
        return np.reshape(dZ, self.input_shape)


class FCLayer:

    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.W = None

    def forward_propagate(self, input):
        if self.W is None:
            self.W = np.random.random((self.num_neurons, input.shape[1] + 1)) * 0.0001
        self.input = np.hstack([input, np.ones((input.shape[0], 1))])  # add bias inputs
        self.Z = np.dot(self.input, self.W.transpose())
        return self.activate(self.Z)

    def backpropagate(self, dA, learning_rate):
        dZ = self.gradient(dA, self.Z)
        dW = np.dot(self.input.transpose(), dZ).transpose() / dA.shape[0]
        dA_prev = np.dot(dZ, self.W)
        dA_prev = np.delete(dA_prev, dA_prev.shape[1] - 1, 1)  # remove bias inputs
        self.W = self.W - learning_rate * dW
        return dA_prev


class FCLayer_ReLU(FCLayer):

    def __init__(self, num_neurons):
        self.activate = lambda Z: np.maximum(0.0, Z)
        self.gradient = lambda dA, Z: dA * (Z > 0.0)
        super().__init__(num_neurons)


class FCLayer_Sigmoid(FCLayer):

    def __init__(self, num_neurons):
        self.activate = lambda Z: 1.0 / (1.0 + np.exp(-Z))
        self.gradient = lambda dA, Z: dA / (1.0 + np.exp(-Z)) * (1.0 - (1.0 / (1.0 + np.exp(-Z))))
        super().__init__(num_neurons)


class FCLayer_Softmax(FCLayer):

    def __init__(self, num_neurons):
        self.activate = lambda Z: np.exp(1.0 / (1.0 + np.exp(-Z))) / np.expand_dims(
            np.sum(np.exp(1.0 / (1.0 + np.exp(-Z))), axis=1), 1)
        self.gradient = lambda dA, Z: dA / (1.0 + np.exp(-Z)) * (1.0 - (1.0 / (1.0 + np.exp(-Z))))
        super().__init__(num_neurons)


class Network:

    def __init__(self, layers=[]):
        self.layers = layers

    def predict(self, X):
        A = np.array(X)
        for i in range(len(self.layers)):
            A = self.layers[i].forward_propagate(A)
        A = np.clip(A, 1e-15, None)  # clip to avoid log(0) in CCE
        A += np.random.random(A.shape) * 0.00001  # small amount of noise to break ties
        return A

    def evaluate(self, X, Y):
        A = self.predict(X)
        Y = np.array(Y)
        cce = -np.sum(Y * np.log(A)) / A.shape[0]  # categorical cross-entropy
        B = np.array(list(1.0 * (A[i] == np.max(A[i])) for i in range(A.shape[0])))
        ce = np.sum(np.abs(B - Y)) / len(Y) / 2.0  # class error
        return (A, cce, ce)

    def train(self, X, Y, learning_rate):
        A, cce, ce = self.evaluate(X, Y)
        dA = A - Y
        for i in reversed(range(len(self.layers))):
            dA = self.layers[i].backpropagate(dA, learning_rate)
        return (np.copy(self.layers), cce, ce)


class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers

    def forward_propagate(self, X):
        A = np.array(X)
        for i in range(len(self.layers)):
            A = self.layers[i].forward_propagate(A)
        A = np.clip(A, 1e-15, None)  # clip to avoid log(0) in CCE
        A += np.random.random(A.shape) * 0.00001  # small amount of noise to break ties
        return A

    def backpropagate(self, dA, learning_rate):
        for i in reversed(range(len(self.layers))):
            dA = self.layers[i].backpropagate(dA, learning_rate)
        return dA
