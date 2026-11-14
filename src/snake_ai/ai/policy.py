"""The policy that picks Snake moves from game states."""

import logging
from pathlib import Path

import torch
import torch.nn as nn

from snake_ai.utils import MODEL_BASE_PATH, TILE_COUNT

from .policy_model import PolicyModel

logger = logging.getLogger(__name__)


class Policy(nn.Module):
    """Wraps a ``PolicyModel`` with state encoding and greedy move selection.

    This is the object the server and ``DQNTrainer`` share: the trainer
    updates its weights, and the server calls ``act`` for each move. Use
    ``load_model`` to get a policy with trained weights.

    Attributes:
        policy_model: The Q-network that scores each action.
        device: ``"cuda"`` if available, otherwise ``"cpu"``. The network is
            moved to this device.
    """

    def __init__(self):
        """Builds a ``PolicyModel`` with random weights on the best device.

        Logs a warning if CUDA is not available.
        """
        super().__init__()
        self.policy_model = PolicyModel(3)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logger.warning("Using %s to train the model", self.device)

        self.policy_model.to(self.device)

    @classmethod
    def load_model(
        cls,
        model_file: str,
        model_path: str | Path = MODEL_BASE_PATH
    ) -> "Policy":
        """Creates a policy and loads saved weights into it.

        The weights are mapped onto the new policy's device, and the network
        is put into eval mode.

        Args:
            model_file: The file name of the saved state dict, e.g.
                ``"model.pt"``.
            model_path: The directory that contains ``model_file``. Defaults
                to ``MODEL_BASE_PATH`` (``policy_models``), which is relative
                to the current working directory.

        Returns:
            A new ``Policy`` with the loaded weights.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)

        agent = cls()
        model_file = model_path / model_file
        agent.policy_model.load_state_dict(
            torch.load(model_file, map_location=agent.device)
        )
        agent.policy_model.eval()
        logger.info("Model loaded")

        return agent

    def forward(self, x):
        """Computes the Q-values for one encoded state or a batch of them.

        Args:
            x: An encoded state of shape ``(N, N, 3)`` or a batch of shape
                ``(B, N, N, 3)``, on ``device``.

        Returns:
            A tensor of shape ``(B, 4)`` with one Q-value per action.
        """
        return self.policy_model(x)

    def get_move(self, state):
        """Returns the greedy action for an encoded state.

        The state is moved to ``device`` first.

        Args:
            state: An encoded state of shape ``(N, N, 3)``.

        Returns:
            The index of the action with the highest Q-value.
        """
        state = state.to(self.device)
        with torch.no_grad():
            q_values = self.policy_model(state)
        action = torch.argmax(q_values, dim=-1).item()
        return action

    @torch.inference_mode()
    def act(self, plain_state):
        """Returns the greedy action for a plain game state.

        This is the inference entry point the server uses.

        Args:
            plain_state: A ``(snake_position, price_position)`` tuple, as
                returned by ``SnakeEnv.get_state``.

        Returns:
            An action index into ``snake.game.ACTIONS``.
        """
        return self.get_move(self.encode_state(plain_state))

    @staticmethod
    def encode_state(plain_state):
        """Turns a plain game state into the network's input tensor.

        Args:
            plain_state: A ``(snake_position, price_position)`` tuple. The
                snake positions are ``[x, y]`` pairs, head first.

        Returns:
            A float tensor of shape ``(N, N, 3)``, indexed as ``[x, y, c]``.
            Channel 0 marks the head, channel 1 the rest of the body and
            channel 2 the food.
        """
        snake_position, price_position = plain_state
        grid_size = TILE_COUNT

        # Initialize the state grid with zeros for each channel
        # We will use 3 channels:
        # Channel 0: Snake's head
        # Channel 1: Snake's body (excluding head)
        # Channel 2: Treasure
        state = torch.zeros(grid_size, grid_size, 3, dtype=torch.float32)

        # Channel 0: Mark the snake's head position
        head_pos = snake_position[0]
        state[head_pos[0], head_pos[1], 0] = 1.0

        # Channel 1: Mark the snake's body positions (excluding head)
        for pos in snake_position[1:]:
            state[pos[0], pos[1], 1] = 1.0

        # Channel 2: Mark the treasure position
        state[price_position[0], price_position[1], 2] = 1.0

        return state
