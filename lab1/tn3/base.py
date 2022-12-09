from abc import ABC, abstractmethod
from typing import List, Union


class Layer(ABC):
    def __init__(self, name):
        self.name = name
        self._params = dict()
        self.__train = True

    def is_train(self):
        return self.__train

    def train(self):
        self.__train = True

    def eval(self):
        self.__train = False

    @abstractmethod
    def forward(self, x):
        pass

    def zero_grad(self):
        pass

    @abstractmethod
    def backward(self, grad_out):
        pass

    def update(self, new_params: dict):
        pass

    @property
    def params(self):
        return self._params


class Module:
    def __init__(self):
        self.layers: List[Layer] = []  # All Layer
        self.graph = dict()

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    @property
    def params(self):
        return [layer.params for layer in self.layers]


class Sequential(Layer):

    def __init__(self, name, layers: List[Layer]):
        super().__init__(name)
        self.layers: List[Layer] = layers

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    @property
    def params(self):
        return [layer.params for layer in self.layers]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def update(self, new_params: dict):
        for layer in self.layers:
            layer.update(new_params)


class OptimBase(ABC):

    def __init__(self, layers: List[Layer], *args, **kwargs):
        self._layers = layers

    def update(self):
        self._update_layers(self._layers)

    def _update_layers(self, layers: List[Union[Layer]]):
        for layer in layers:
            if isinstance(layer, Sequential):
                self._update_layers(layer.layers)
            elif isinstance(layer, Layer):
                self._update_layer(layer)

    @abstractmethod
    def _update_layer(self, layer: Layer):
        pass

    def zero_grad(self):
        pass
