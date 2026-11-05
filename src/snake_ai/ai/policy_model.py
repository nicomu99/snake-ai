import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x: "torch.Tensor"):
        # input is (batch_size, height, width, channels)
        # change first

        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = x.permute(0, 3, 1, 2)
        for _, layer in enumerate(self.sequential):
            x = layer(x)
        return x
        # return self.sequential(x)
