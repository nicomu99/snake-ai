"""The policy that picks Snake moves from game states."""

import logging
from pathlib import Path

import torch
import torch.nn as nn

from snake_ai.ai.policy_model import PolicyModel
from snake_ai.utils import TILE_COUNT

logger = logging.getLogger(__name__)


# Direction channels of the encoded state. Their order matches
# ``snake.snake_env.ACTIONS``, so channel ``c`` belongs to action ``c - 3``.
_DIR_TO_CHANNEL = {"UP": 3, "DOWN": 4, "LEFT": 5, "RIGHT": 6}
# The opposite action of each action index (UP <-> DOWN, LEFT <-> RIGHT).
_REVERSE = [1, 0, 3, 2]


class Policy(nn.Module):
    """Wraps a ``PolicyModel`` with state encoding and greedy move selection.

    This is the object the server and ``DQNTrainer`` share: the trainer
    updates its weights, and the server calls ``act`` for each move. Use
    ``load_model`` to get a policy with trained weights.

    Reversing the snake's direction loses the game, so the greedy choice
    never picks it: ``get_move`` masks it with ``reversal_mask``. ``forward``
    returns the raw, unmasked Q-values.

    Attributes:
        policy_model: The Q-network that scores each action.
        device: ``"cuda"`` if available, otherwise ``"cpu"``. The network is
            moved to this device.
    """

    _MODEL_CHANNELS = 7

    def __init__(self):
        """Builds a ``PolicyModel`` with random weights on the best device.

        Logs a warning if CUDA is not available.
        """
        super().__init__()
        self.policy_model = PolicyModel(self._MODEL_CHANNELS)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logger.warning("Using %s to train the model", self.device)

        self.policy_model.to(self.device)

    @classmethod
    def load_model(
        cls,
        model_file: str,
        model_path: str | Path,
    ) -> "Policy":
        """Creates a policy and loads saved weights into it.

        The weights are mapped onto the new policy's device, and the network
        is put into eval mode.

        Args:
            model_file: The file name of the saved state dict, e.g.
                ``"model.pt"``, relative to ``model_path``. An absolute path
                is used as is and ``model_path`` is ignored.
            model_path: The directory that contains ``model_file``.

        Returns:
            A new ``Policy`` with the loaded weights.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)

        agent = cls()
        model_file = model_path / model_file
        agent.load_state_dict(
            torch.load(model_file, map_location=agent.device)
        )
        agent.policy_model.eval()
        logger.info("Model loaded")

        return agent

    def forward(self, x):
        """Computes the Q-values for one encoded state or a batch of them.

        Args:
            x: An encoded state of shape ``(N, N, 7)`` or a batch of shape
                ``(B, N, N, 7)``, on ``device``.

        Returns:
            A tensor of shape ``(B, 4)`` with one Q-value per action. The
            reversal is not masked; see ``reversal_mask``.
        """
        return self.policy_model(x)

    def get_move(self, state):
        """Returns the greedy action for an encoded state.

        The state is moved to ``device`` first. The action that would reverse
        the snake's direction is masked, so it is never returned.

        Args:
            state: An encoded state of shape ``(N, N, 7)``.

        Returns:
            The index of the allowed action with the highest Q-value.
        """
        state = state.to(self.device)
        with torch.no_grad():
            q_values = self.policy_model(state)

        # Mask negative direction
        q_values = q_values.masked_fill(
            self.reversal_mask(state).reshape(q_values.shape), float("-inf"))

        action = torch.argmax(q_values, dim=-1).item()
        return action

    @torch.inference_mode()
    def act(self, plain_state):
        """Returns the greedy action for a plain game state.

        This is the inference entry point the server uses.

        Args:
            plain_state: A ``(snake_position, price_position, snake_direction)`` tuple,
                as returned by ``SnakeEnv.get_state``.

        Returns:
            An action index into ``snake.snake_env.ACTIONS``.
        """
        return self.get_move(self.encode_state(plain_state))

    @staticmethod
    def reversal_mask(state):
        """Marks the action that would reverse the snake's direction.

        The direction is read from channels 3-6 of the encoded state, which
        are the same on every tile. Before the first move no direction is
        set, so nothing is masked.

        Args:
            state: An encoded state of shape ``(N, N, 7)`` or a batch of
                shape ``(B, N, N, 7)``.

        Returns:
            A bool tensor of shape ``(4,)`` or ``(B, 4)`` in the order of
            ``snake.snake_env.ACTIONS``, True for the reversing action.
        """
        direction = state[..., 0, 0, 3:7] > 0.5
        return direction[..., _REVERSE]

    @staticmethod
    def encode_state(plain_state):
        """Turns a plain game state into the network's input tensor.

        Args:
            plain_state: A ``(snake_position, price_position, snake_direction)`` tuple.
                The snake positions are ``[x, y]`` pairs, head first.

        Returns:
            A float tensor of shape ``(N, N, 7)``, indexed as ``[x, y, c]``.
            Channel 0 marks the head, channel 1 the rest of the body,
            channel 2 the food, and the remaining channels show the snake's direction.
        """
        snake_position, price_position, snake_direction = plain_state
        grid_size = TILE_COUNT

        # Initialize the state grid with zeros for each channel
        # We will use 7 channels:
        # Channel 0: Snake's head
        # Channel 1: Snake's body (excluding head)
        # Channel 2: Treasure
        # Channels 3-6: Snake's direction with one-hot encoding
        # each channel represents one direction; if the snake is currently moving
        # into the channel's direction, the values are 1, else 0.
        # 3: UP, 4: DOWN, 5: LEFT, 6: RIGHT
        state = torch.zeros(
            grid_size, grid_size, Policy._MODEL_CHANNELS, dtype=torch.float32)

        # Channel 0: Mark the snake's head position
        head_pos = snake_position[0]
        state[head_pos[0], head_pos[1], 0] = 1.0

        # Channel 1: Mark the snake's body positions (excluding head)
        for pos in snake_position[1:]:
            state[pos[0], pos[1], 1] = 1.0

        # Channel 2: Mark the treasure position
        state[price_position[0], price_position[1], 2] = 1.0

        if snake_direction is not None:
            state[:, :, _DIR_TO_CHANNEL[snake_direction]] = 1

        return state
