"""The convolutional Q-network used by the agent."""

import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    """A small CNN that predicts one Q-value per action from a board state.

    Two unpadded convolutions shrink a 5x5 board to a 64x3x1 feature map,
    which is flattened and passed through two fully connected layers. The
    ``Linear(192, 128)`` input size only works for a 5x5 board (see
    ``utils.constants.TILE_COUNT``).

    The four outputs are in the order of ``snake.game.ACTIONS``:
    ``UP``, ``DOWN``, ``LEFT``, ``RIGHT``.

    Args:
        in_channels: The number of channels per tile in the encoded state
            (3 for head, body and food, see ``Agent.encode_state``).
    """

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
        """Computes the Q-values for one state or a batch of states.

        Args:
            x: A channels-last state tensor of shape ``(H, W, C)`` or a batch
                of shape ``(B, H, W, C)``. A single state is treated as a
                batch of one.

        Returns:
            A tensor of shape ``(B, 4)`` with one Q-value per action.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        return self.sequential(x)
