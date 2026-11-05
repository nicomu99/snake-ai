import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x: "torch.Tensor"):
        return self.sequential(x)
