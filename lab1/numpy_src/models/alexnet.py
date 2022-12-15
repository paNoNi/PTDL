import numpy as np

from lab1.numpy_src.base import Module, Sequential
from lab1.numpy_src.layers import Conv2D, MaxPooling, Flatten, FC, Dropout, BatchNormLayer, BatchNorm2D
from lab1.numpy_src.activations import Softmax, Relu

class AlexNet(Module):
    def __init__(self):
        super().__init__()
        self.block_left_first = self.get_first_block(subname='left')

        self.block_left_second = self.get_second_block(subname='left')

        self.block_left_third = self.get_third_block(subname='left')

        self.block_left_forth = self.get_forth_block(subname='left')

        self.block_firth = self.get_firth_block(subname='left')

        self.layers = [
            self.block_left_first,
            self.block_left_second,
            self.block_left_third,
            self.block_left_forth,
            self.block_firth
        ]


    def forward(self, x):
        x_left = self.block_left_first(np.copy(x))
        x_left = self.block_left_second(np.copy(x_left))
        x_left = self.block_left_third(np.copy(x_left))
        x_left = self.block_left_forth(np.copy(x_left))
        x_full = self.block_firth(np.copy(x_left))
        return np.copy(x_full)

    def backward(self, grad):
        grad = self.block_firth.backward(np.copy(grad))
        grad_left = self.block_left_forth.backward(np.copy(grad))
        grad_left = self.block_left_third.backward(np.copy(grad_left))
        grad_left = self.block_left_second.backward(np.copy(grad_left))
        grad_left = self.block_left_first.backward(np.copy(grad_left))



    def get_first_block(self, subname: str):
        return Sequential(name=f'block_{subname}_first_stage', layers=[
            Conv2D(name=f'conv1_{subname}_first_stage', in_channels=3, out_channels=96, stride=4, kernel_size=11,
                   padding=0),
            BatchNorm2D(name=f'bn2d1_{subname}_first_stage', n_channels=96),
            Relu(name=f'relu1_{subname}_first_stage'),
            MaxPooling(name=f'pooling1_{subname}_first_stage', ksize=3, stride=2),
            Conv2D(name=f'conv2_{subname}_first_stage', in_channels=96, out_channels=256, kernel_size=5, padding=2,
                   stride=1),
            BatchNorm2D(name=f'bn2d2_{subname}_first_stage', n_channels=256),
            Relu(name=f'relu2_{subname}_first_stage'),
            MaxPooling(name=f'pooling2_{subname}_first_stage', ksize=3, stride=2),
        ])

    def get_second_block(self, subname: str):
        return Sequential(name=f'block_{subname}_second_stage', layers=[
            Conv2D(name=f'conv1_{subname}_second_stage', in_channels=256, out_channels=384, kernel_size=3, stride=1,
                   padding=1),
            BatchNorm2D(name=f'bn2d1_{subname}_second_stage', n_channels=384),
            Relu(name=f'relu1_{subname}_second_stage'),
            Conv2D(name=f'conv2_{subname}_second_stage', in_channels=384, out_channels=384, kernel_size=3, padding=1,
                   stride=1),
            BatchNorm2D(name=f'bn2d2_{subname}_second_stage', n_channels=384),
            Relu(name=f'relu2_{subname}_second_stage'),
            Conv2D(name=f'conv3_{subname}_second_stage', in_channels=384, out_channels=256, kernel_size=3, padding=1,
                   stride=1),
            BatchNorm2D(name=f'bn2d3_{subname}_second_stage', n_channels=256),
            Relu(name=f'relu3_{subname}_second_stage'),
            MaxPooling(name=f'pooling1_{subname}_second_stage', ksize=3, stride=2),
            Flatten(name='flat'),
        ])


    def get_third_block(self, subname: str):
        return Sequential(name=f'block_{subname}_third_stage', layers=[
            # Dropout('dropout2', prob=0.5),
            FC(name=f'FC_[{subname}_third_stage', in_channels=9216, out_channels=4096),
            Relu(name=f'relu_{subname}_third_stage'),
        ])

    def get_forth_block(self, subname: str):
        return Sequential(name=f'block_{subname}_forth_stage', layers=[
            # Dropout('dropout2', prob=0.5),
            FC(name=f'FC_[{subname}_forth_stage', in_channels=4096, out_channels=4096),
            Relu(name=f'relu_{subname}_forth_stage'),
        ])

    def get_firth_block(self, subname: str):
        return Sequential(name=f'block_{subname}_firth_stage', layers=[
            FC(name=f'FC_[{subname}_firth_stage', in_channels=4096, out_channels=10),
            BatchNormLayer(name=f'bn_{subname}_firth_stage', dims=10),
            Softmax(name=f'softmax_{subname}_firth_stage')
        ])


# class AlexNet(Module):
#     def __init__(self):
#         super().__init__()
#         self.block_left_first = self.get_first_block(subname='left')
#         self.block_right_first = self.get_first_block(subname='right')
#
#         self.block_left_second = self.get_second_block(subname='left')
#         self.block_right_second = self.get_second_block(subname='right')
#
#         self.block_left_third = self.get_third_block(subname='left')
#         self.block_right_third = self.get_third_block(subname='right')
#
#         self.block_left_forth = self.get_forth_block(subname='left')
#         self.block_right_forth = self.get_forth_block(subname='right')
#
#         self.block_firth = self.get_firth_block(subname='left')
#
#         self.layers = [
#             self.block_left_first,
#             self.block_right_first,
#             self.block_left_second,
#             self.block_right_second,
#             self.block_left_third,
#             self.block_right_third,
#             self.block_left_forth,
#             self.block_right_forth,
#             self.block_firth
#         ]
#
#
#     def forward(self, x):
#         x_left = self.block_left_first(np.copy(x))
#         x_right = self.block_right_first(np.copy(x))
#         x_full = x_left + x_right
#         x_left = self.block_left_second(np.copy(x_full))
#         x_right = self.block_right_second(np.copy(x_full))
#         x_full = x_left + x_right
#         x_left = self.block_left_third(np.copy(x_full))
#         x_right = self.block_right_third(np.copy(x_full))
#         x_full = x_left + x_right
#         x_left = self.block_left_forth(np.copy(x_full))
#         x_right = self.block_right_forth(np.copy(x_full))
#         x_full = x_left + x_right
#         x_full = self.block_firth(np.copy(x_full))
#         return np.copy(x_full)
#
#     def backward(self, grad):
#         grad = self.block_firth.backward(np.copy(grad))
#         grad_left = self.block_left_forth.backward(np.copy(grad))
#         grad_left = self.block_left_third.backward(np.copy(grad_left))
#         grad_left = self.block_left_second.backward(np.copy(grad_left))
#         grad_left = self.block_left_first.backward(np.copy(grad_left))
#
#         grad_right = self.block_right_forth.backward(np.copy(grad))
#         grad_right = self.block_right_third.backward(np.copy(grad_right))
#         grad_right = self.block_right_second.backward(np.copy(grad_right))
#         grad_right = self.block_right_first.backward(np.copy(grad_right))
#
#
#     def get_first_block(self, subname: str):
#         return Sequential(name=f'block_{subname}_first_stage', layers=[
#             Conv2D(name=f'conv1_{subname}_first_stage', in_channels=3, out_channels=48, stride=4, kernel_size=11,
#                    padding=0),
#             Relu(name=f'relu1_{subname}_first_stage'),
#             MaxPooling(name=f'pooling1_{subname}_first_stage', ksize=3, stride=2),
#             Conv2D(name=f'conv2_{subname}_first_stage', in_channels=48, out_channels=128, kernel_size=5, padding=2,
#                    stride=1),
#             Relu(name=f'relu2_{subname}_first_stage'),
#             MaxPooling(name=f'pooling1_{subname}_first_stage', ksize=3, stride=2),
#         ])
#
#     def get_second_block(self, subname: str):
#         return Sequential(name=f'block_{subname}_first_stage', layers=[
#             Conv2D(name=f'conv1_{subname}_second_stage', in_channels=128, out_channels=192, kernel_size=3, stride=1,
#                    padding=1),
#             Relu(name=f'relu1_{subname}_second_stage'),
#             Conv2D(name=f'conv2_{subname}_second_stage', in_channels=192, out_channels=192, kernel_size=3, padding=1,
#                    stride=1),
#             Relu(name=f'relu2_{subname}_second_stage'),
#             Conv2D(name=f'conv2_{subname}_second_stage', in_channels=192, out_channels=128, kernel_size=3, padding=1,
#                    stride=1),
#             Relu(name=f'relu2_{subname}_second_stage'),
#             MaxPooling(name=f'pooling1_{subname}_second_stage', ksize=3, stride=2),
#             Flatten(name='flat'),
#         ])
#
#
#     def get_third_block(self, subname: str):
#         return Sequential(name=f'block_{subname}_forth_stage', layers=[
#             BatchNormLayer(dims=4608, name='b1'),
#             FC(name=f'FC_[{subname}_forth_stage', in_channels=4608, out_channels=2048),
#             Relu(name=f'relu_{subname}_forth_stage'),
#         ])
#
#     def get_forth_block(self, subname: str):
#         return Sequential(name=f'block_{subname}_firth_stage', layers=[
#             BatchNormLayer(dims=2048, name='b2'),
#             FC(name=f'FC_[{subname}_firth_stage', in_channels=2048, out_channels=2048),
#             Relu(name=f'relu_{subname}_firth_stage'),
#         ])
#
#     def get_firth_block(self, subname: str):
#         return Sequential(name=f'block_{subname}_sixth_stage', layers=[
#             BatchNormLayer(dims=2048, name='b3'),
#             FC(name=f'FC_[{subname}_sixth_stage', in_channels=2048, out_channels=10),
#             Softmax(name=f'softmax_{subname}_sixth_stage')
#         ])
