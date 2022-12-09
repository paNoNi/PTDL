import numpy as np


class SGD:
    """
    Implements vanilla SGD update
    """

    def __init__(self, params, lr: float = 0.01):
        self.__params: list = params
        self.lr = lr

    def update(self):
        for params in self.__params:
            for param_name in list(params.keys()):
                params[param_name].value = self.__update_one(params[param_name])

    def __update_one(self, params):
        """
        Performs SGD update
        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate
        Returns:
        updated_weights, np array same shape as w
        """
        # print(params.value.max(), params.grad.max(), self.lr)
        return params.value - params.grad * self.lr
