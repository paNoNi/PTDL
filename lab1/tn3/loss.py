import numpy as np
from torch import nn


def Softmax(input):
    vec_max = np.max(input, axis=1, keepdims=True)
    input -= vec_max
    exp = np.exp(input)
    softmax_pro = exp / np.sum(exp, axis=1, keepdims=True)
    return softmax_pro


class CrossEntropyLoss:
    def __init__(self, reduce='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduce = reduce

    def __call__(self, pred: np.ndarray, label: np.ndarray):
        self.softmax_p = pred
        self.real = label
        loss = 0
        for i in range(label.shape[0]):
            loss += -np.log(self.softmax_p[i, label[i]] + 1e-12)

        loss /= pred.shape[0]
        grad = self.grad()
        conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        return loss, grad

    def grad(self):
        grad = self.softmax_p.copy()
        for i in range(self.real.shape[0]):
            grad[i, self.real[i]] -= 1
        return grad


class MSELoss:
    def __init__(self, reduce='mean'):
        super(MSELoss, self).__init__()
        self.reduce = reduce

    def __call__(self, pred, label):
        assert pred.shape == label.shape, 'pred and gt shape must be same'
        loss = np.sum(np.square((pred - label)), axis=-1)
        if self.reduce == 'mean':
            loss = np.mean(loss)
        else:
            loss = np.sum(loss)
        grad = (pred - label)
        return loss, grad
