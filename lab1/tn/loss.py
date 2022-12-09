import numpy as np


class Loss:
    def __init__(self):
        pass


def get_target_neurons(target_index, size):
    px = np.zeros(size)
    if (type(target_index) == int):
        px[target_index] = 1
    else:
        for row, i in enumerate(target_index):
            px[row][i] = 1

    return px


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def calculate(self, probs, target_index):
        px = get_target_neurons(target_index, probs.shape)
        px = np.zeros_like(probs)
        px[list(range(px.shape[0])), target_index] = 1

        return -1 * np.sum(px * np.log(probs))


class MSE(Loss):
    def __init__(self):
        super().__init__()
        self.__last_values = (0, 0)

    def calculate(self, true_values: np.ndarray, predictions: np.ndarray):
        self.__last_values = true_values, predictions
        return np.sum(np.power(true_values - predictions, 2)) / true_values.shape[0]

    def grad(self):
        return self.__last_values[0] - self.__last_values[1]
