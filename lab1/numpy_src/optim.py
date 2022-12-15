from typing import List

import numpy as np

from lab1.numpy_src.base import OptimBase, Layer


class SGD(OptimBase):
    def __init__(self, layers: List[Layer],
                 lr: float = 0.01):
        super(SGD, self).__init__(layers)
        self.lr = lr

    def _update_layer(self, layer: Layer):
        new_params = dict()
        cur_params = layer.params
        for key in list(cur_params.keys()):
            new_params[key] = dict()
            value = self._update_one(cur_params[key])
            new_params[key]['value'] = value
            new_params[key]['grad'] = cur_params[key]['grad']
        layer.update(new_params)

    def _update_one(self, params):
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
        return params['value'] - params['grad'] * self.lr


class AdaSmooth(OptimBase):
    def __init__(self, layers: List[Layer],
                 lr: float = 0.01,
                 p_fast: float = 0.5,
                 p_slow: float = 0.9,
                 epsilon=1e-12,
                 window_size: int = 5):
        super(AdaSmooth, self).__init__(layers)
        self.state = dict()
        self.state['weights_hist'] = dict()
        self.state['Egs'] = dict()
        self.state['delta_data'] = dict()
        self.const = {'lr': lr, 'window': window_size, 'eps': epsilon, 'p1': p_fast, 'p2': p_slow}

    def _update_layer(self, layer: Layer):
        cur_params = layer.params
        layer_name = layer.name

        if len(list(cur_params.keys())) != 0 and layer_name not in list(self.state['weights_hist'].keys()):
            self.state['weights_hist'][layer_name] = dict()
            self.state['delta_data'][layer_name] = dict()
            self.state['Egs'][layer_name] = dict()

        for i, key in enumerate(list(cur_params.keys())):

            if key not in list(self.state['delta_data'][layer_name].keys()):
                self.state['weights_hist'][layer_name][key] = None
                self.state['delta_data'][layer_name][key] = None
                self.state['Egs'][layer_name][key] = 0

            cur_params[key]['value'] = self.__update_one(cur_params[key], layer_name, key)

        layer.update(cur_params)

    def __update_one(self, params, layer_name: str, key: str):

        grad = params['grad']
        weights = params['value']

        if self.state['delta_data'][layer_name][key] is None or self.state['delta_data'][layer_name][key].shape[0] < self.const['window']:
            delta_data = params['grad'] * self.const['lr']
        else:

            abs_sum_x = np.abs(weights - self.state['weights_hist'][layer_name][key][0])
            sum_abs_x = np.sum(np.abs(self.state['delta_data'][layer_name][key]), axis=0)

            e = np.divide(abs_sum_x, sum_abs_x + self.const['eps'])
            c2 = np.power((self.const['p2'] - self.const['p1']) * e + (1 - self.const['p2']), 2)
            self.state['Egs'][layer_name][key] = np.multiply(c2, np.power(grad, 2)) + np.multiply((1 - c2), self.state['Egs'][layer_name][key])

            lregs = self.const['lr'] / (np.sqrt(self.state['Egs'][layer_name][key] + self.const['eps']))

            delta_data = np.multiply(grad, lregs)

        if self.state['weights_hist'][layer_name][key] is None:
            self.state['weights_hist'][layer_name][key] = np.expand_dims(weights, axis=0)
            self.state['delta_data'][layer_name][key] = np.expand_dims(delta_data, axis=0)
        else:
            self.state['weights_hist'][layer_name][key] = np.concatenate([self.state['weights_hist'][layer_name][key], np.expand_dims(weights, axis=0)], axis=0)
            self.state['delta_data'][layer_name][key] = np.concatenate([self.state['delta_data'][layer_name][key], np.expand_dims(delta_data, axis=0)], axis=0)

        self.state['weights_hist'][layer_name][key] = self.state['weights_hist'][layer_name][key][-self.const['window']:]
        self.state['delta_data'][layer_name][key] = self.state['delta_data'][layer_name][key][-self.const['window']:]

        return params['value'] - delta_data
