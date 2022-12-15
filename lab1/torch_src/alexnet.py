import torch
from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class fasion_mnist_alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out


class HalfAlexNet(nn.Module):

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
            nn.Conv2d(in_channels=3, out_channels=48, stride=4, kernel_size=11,
                      padding=0),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def get_second_block(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )

    def get_third_block(self):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4608, out_features=2048),
            nn.ReLU(),
        )

    def get_forth_block(self):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
        )

    def get_firth_block(self):
        return nn.Sequential(
            nn.Linear(in_features=1024, out_features=196),
            nn.Softmax(dim=1)
        )


class AlexNet_(nn.Module):

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
        x = F.softmax(x, dim=1)
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
            # Dropout('dropout2', prob=0.5),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
        )

    def get_forth_block(self):
        return nn.Sequential(
            # Dropout('dropout2', prob=0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
        )

    def get_firth_block(self):
        return nn.Sequential(
            nn.Linear(in_features=2048, out_features=196)
        )
