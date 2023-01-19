import torch
from torch import nn
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights


class SwinModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        self.swin.head = torch.nn.Identity()

    def forward(self, x):
        return self.swin(x)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_1 = self.block(in_features=1024, out_features=512)
        self.block_2 = self.block(in_features=512, out_features=512)
        self.block_3 = self.block(in_features=512, out_features=256)
        self.block_4 = self.block(in_features=256, out_features=128)
        self.block_5 = self.block(in_features=128, out_features=64)
        self.head = nn.Sequential(
            nn.Linear(in_features=64, out_features=4),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.head(x)
        return x

    def block(self, in_features: int, out_features: int):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=out_features),
        )
