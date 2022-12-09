from typing import List

import numpy as np


from lab1.tn3.base import OptimBase, Layer


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
                 epsilon=1e-12):
        super(AdaSmooth, self).__init__(layers)
        self.t = 0
        self.lr = lr
        self.p_fast = p_fast
        self.p_slow = p_slow
        self.eps = epsilon
        self.weights_hist = dict()
        self.Egs = dict()
        self.cur_window = 10

    # def update(self):
    #     for layer in self._layers:
    #         new_params = dict()
    #         cur_params = layer.params
    #         layer_name = layer.name
    #
    #         if len(list(cur_params.keys())) != 0 and layer_name not in list(self.weights_hist.keys()):
    #             self.weights_hist[layer_name] = dict()
    #
    #         for i, key in enumerate(list(cur_params.keys())):
    #             new_params[key] = dict()
    #
    #             if key not in list(self.weights_hist[layer_name].keys()):
    #                 self.weights_hist[layer_name][key] = np.zeros((1, *cur_params[key]['value'].shape))
    #
    #             value = self.__update_one(cur_params[key], layer_name, key)
    #             new_params[key]['value'] = value
    #             new_params[key]['grad'] = cur_params[key]['grad']
    #             self.weights_hist[layer_name][key] = \
    #                 np.concatenate([self.weights_hist[layer_name][key],
    #                                 np.expand_dims(cur_params[key]['value'], axis=0)], axis=0)
    #             self.weights_hist[layer_name][key] = self.weights_hist[layer_name][key][-self.cur_window:]
    #         layer.update(new_params)

    def _update_layer(self, layer: Layer):
        new_params = dict()
        cur_params = layer.params
        layer_name = layer.name

        if len(list(cur_params.keys())) != 0 and layer_name not in list(self.weights_hist.keys()):
            self.weights_hist[layer_name] = dict()

        for i, key in enumerate(list(cur_params.keys())):
            new_params[key] = dict()

            if key not in list(self.weights_hist[layer_name].keys()):
                self.weights_hist[layer_name][key] = np.zeros((1, *cur_params[key]['value'].shape))

            value = self.__update_one(cur_params[key], layer_name, key)
            new_params[key]['value'] = value
            new_params[key]['grad'] = cur_params[key]['grad']
            self.weights_hist[layer_name][key] = \
                np.concatenate([self.weights_hist[layer_name][key],
                                np.expand_dims(cur_params[key]['value'], axis=0)], axis=0)
            self.weights_hist[layer_name][key] = self.weights_hist[layer_name][key][-self.cur_window:]
        layer.update(new_params)

    def __update_one(self, params, layer_name: str, key: str):

        if layer_name not in list(self.Egs.keys()):
            self.Egs[layer_name] = dict()
            # print(self.Egs.keys())

        if key not in list(self.Egs[layer_name].keys()):
            # self.Egs[layer_name][key] = np.power(params['grad'], 2)
            self.Egs[layer_name][key] = 0
            # print(self.Egs[layer_name][key].mean())

        abs_sum_x = np.abs(np.sum(self.weights_hist[layer_name][key] - np.expand_dims(params['value'], axis=0), axis=0))
        sum_abs_x = np.sum(np.abs(self.weights_hist[layer_name][key] - np.expand_dims(params['value'], axis=0)), axis=0)

        sum_abs_x = sum_abs_x + self.eps


        # print('=' * 30)
        # print(abs_sum_x, sum_abs_x)
        e = np.divide(abs_sum_x, sum_abs_x)
        c2 = np.power((self.p_slow - self.p_fast) * e + (1 - self.p_slow), 2)
        self.Egs[layer_name][key] = np.multiply(e, np.power(params['grad'], 2)) + \
                                    np.multiply((1 - c2), self.Egs[layer_name][key])

        # self.Egs[layer_name][key] = np.nan_to_num(self.Egs[layer_name][key])
        # print(self.Egs[layer_name][key].mean())
        # apply lr
        # print(f'EGS: {self.Egs[layer_name][key].min()}')
        # print(f'EGS + eps: {(self.Egs[layer_name][key] + self.eps).min()}')
        # print(f'EGS + eps + sqrt: {(np.sqrt(self.Egs[layer_name][key] + self.eps)).mean()}')
        # print(f'Matrix: {np.multiply(params["grad"], self.lr / (np.sqrt(self.Egs[layer_name][key] + self.eps))).mean()}')
        # print(np.multiply(params['grad'], self.lr / (np.sqrt(self.Egs[layer_name][key] + self.eps))).mean())
        # print('-' * 30)
        # print(np.multiply(params['grad'], self.lr / (np.sqrt(self.Egs[layer_name][key] + self.eps))))
        # print('=' * 30)
        return params['value'] - np.multiply(params['grad'], self.lr / (np.sqrt(self.Egs[layer_name][key] + self.eps)))
