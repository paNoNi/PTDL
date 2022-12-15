import torch
from torch import nn

class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_left_first = self.get_first_block()
        self.block_left_second = self.get_second_block()
        self.block_left_third = self.get_third_block()
        self.block_left_forth = self.get_forth_block()
        self.block_firth = self.get_firth_block()

    def forward(self, x):
        x = self.block_left_first(x)
        x = self.block_left_second(x)
        x = self.block_left_third(x)
        x = self.block_left_forth(x)
        x = self.block_firth(x)
        return x

    def get_first_block(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, stride=4, kernel_size=11,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def get_second_block(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )

    def get_third_block(self):
        return nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=9216, out_features=2048),
            nn.ReLU(),
        )

    def get_forth_block(self):
        return nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
        )

    def get_firth_block(self):
        return nn.Sequential(
            nn.Linear(in_features=2048, out_features=196)
        )
