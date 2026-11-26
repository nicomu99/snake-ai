"""The convolutional Q-network that ``Policy`` wraps."""

import torch
import torch.nn as nn

from snake_ai.utils import TILE_COUNT


class PolicyModel(nn.Module):
    """A small CNN that predicts one Q-value per action from a board state.

    Two 3x3 convolutions with circular padding keep the board size, so the
    feature map is 64 x ``TILE_COUNT`` x ``TILE_COUNT``. Circular padding
    matches the board, which wraps at the edges. The feature map is
    flattened and passed through two fully connected layers. The first
    linear layer's input size is derived from ``utils.constants.TILE_COUNT``,
    so saved weights only fit a model built with the same ``TILE_COUNT``.

    The four outputs are in the order of ``snake.snake_env.ACTIONS``:
    ``UP``, ``DOWN``, ``LEFT``, ``RIGHT``.

    Args:
        in_channels: The number of channels per tile in the encoded state
            (7: head, body, food and four one-hot direction channels, see
            ``Policy.encode_state``).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3),
                      padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3),
                      padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * TILE_COUNT * TILE_COUNT, 128),
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
