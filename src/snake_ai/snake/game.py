"""A headless Snake game that an agent can play.

Attributes:
    ACTIONS (tuple[str, ...]): Maps the agent's action indices to directions.
        The order must match the outputs of ``PolicyModel``.
"""

import copy
import logging

from snake_ai.ai.policy import Policy
from snake_ai.snake.snake_env import SnakeEnv

logger = logging.getLogger(__name__)

# Index = action int produced by the agent
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class Game:
    """A headless game core that knows nothing about how it is displayed.

    ``Game`` wraps a ``SnakeEnv``, lets a ``Policy`` choose moves, counts
    steps and notifies subscribed listeners with a serializable snapshot
    after every move and reset. Frontends (the WebSocket server, the Tk GUI)
    build on top of it.

    ``Game`` also provides ``reset()`` and ``step(action)``, so it can be
    passed to ``DQNTrainer.train`` as the training environment.

    Args:
        policy: The policy that picks moves in ``agent_step``. It can be
            shared between games.

    Attributes:
        snake_env: The underlying game state and rules.
        policy: The policy used by ``agent_step``.
        step_count: The number of moves since the last reset. Moves made
            after the game has ended are not counted.
        game_finished: Whether the current game has been lost or won.
    """

    def __init__(self, policy: Policy):
        self.game_finished = False
        self.snake_env = SnakeEnv()
        self.policy = policy
        self._listeners = []
        self.step_count = 0

    def subscribe(self, listener):
        """Registers a listener that is called after every move and reset.

        Args:
            listener: A callable that takes the dict returned by ``snapshot``.
        """
        self._listeners.append(listener)

    def reset(self, seed: int | None = None):
        """Starts a new game, resets the step counter and notifies listeners.

        Args:
            seed: The random seed for the new game. If None, the current random number
                generator keeps going without being reseeded. Defaults to None.

        Returns:
            The plain state of the new game, see ``SnakeEnv.get_state``.
        """
        self.game_finished = False
        self.step_count = 0
        state = self.snake_env.reset(seed)
        self._notify()
        return state

    def step(self, action):
        """Performs one move and notifies the listeners.

        Once the game is lost or won, further moves leave the board as it
        is, return a reward of 0 and don't increase ``step_count``. The
        listeners are still notified.

        Args:
            action: A direction string (``"UP"``, ``"DOWN"``, ``"LEFT"``,
                ``"RIGHT"``) or an index into ``ACTIONS``.

        Returns:
            A ``(reward, state, lost, won)`` tuple, see
            ``SnakeEnv.perform_move``.

        Raises:
            ValueError: If ``action`` is not a valid direction or index.
        """
        if isinstance(action, int):
            if action < 0 or action > 3:
                raise ValueError(
                    "Invalid action value. Must be between 0 and 3 inclusive.")
            direction = ACTIONS[action] if isinstance(action, int) else action
        else:
            if action not in ACTIONS:
                raise ValueError(f"Invalid action value. Must be in {ACTIONS}")
            direction = action
        logger.debug("Performing move into direction %s", direction)

        reward, state, lost, won = self.snake_env.perform_move(direction)
        if not self.game_finished:
            self.step_count += 1
        if lost or won:
            self.game_finished = True
        self._notify()
        return reward, state, lost, won

    def agent_step(self):
        """Lets the policy choose one greedy move and performs it.

        Returns:
            A ``(reward, state, lost, won)`` tuple, see
            ``SnakeEnv.perform_move``.
        """
        action = self.policy.act(self.snake_env.get_state())
        return self.step(action)

    def snapshot(self):
        """Returns a plain, JSON-serializable view of the current game.

        Returns:
            A dict with ``snake`` (a list of ``[x, y]`` positions, head
            first), ``food`` (``[x, y]``), ``score``, ``lost``, ``won``,
            ``game_state`` (see ``SnakeEnv.get_game_state``) and
            ``step_count``.
        """
        return {
            "snake": [list(pos) for pos in self.snake_env.snake_position],
            "food": list(self.snake_env.price_position),
            "score": self.snake_env.score,
            "lost": self.snake_env.lost,
            "won": self.snake_env.won,
            "game_state": self.snake_env.get_game_state(),
            "step_count": self.step_count,
        }

    def _notify(self):
        """Calls every listener with a fresh snapshot."""
        snapshot = self.snapshot()
        if not self._listeners:
            return
        for listener in self._listeners:
            listener(snapshot)

    def to_dict(self) -> dict:
        """Returns the game as a JSON-serializable dict for persistence.

        The dict holds everything ``from_dict`` needs to continue the game
        exactly, including the food placement. All values are copies, so
        later moves don't change the dict.

        Returns:
            The keys of ``snapshot``, plus ``direction`` (the last move's
            direction, or ``None`` before the first move),
            ``last_snake_position`` (the tail position before the last move)
            and ``rng_state`` (``SnakeEnv.rng.getstate()`` as a list).
        """
        game_state = self.snapshot()
        game_state["direction"] = self.snake_env.direction
        game_state["last_snake_position"] = list(self.snake_env.last_snake_position)

        version, internal, gauss = self.snake_env.rng.getstate()
        game_state["rng_state"] = [version, list(internal), gauss]

        return game_state

    @classmethod
    def from_dict(cls, game_state: dict, policy: Policy) -> "Game":
        """Restores a game from a dict made by ``to_dict``.

        The dict may have gone through JSON. Its values are copied, so the
        dict and the restored game don't share lists. ``game_finished`` is
        derived from ``lost`` and ``won``, and the ``game_state`` key is
        ignored. Listeners are not restored, so callers have to subscribe
        again.

        Args:
            game_state: A dict returned by ``to_dict``.
            policy: The policy that picks moves in the restored game.

        Returns:
            A new ``Game`` that continues where the saved one stopped,
            including the food placement.

        Raises:
            KeyError: If a key written by ``to_dict`` is missing.
        """
        game = cls(policy)
        game.step_count = game_state["step_count"]
        game.snake_env.snake_position = copy.deepcopy(game_state["snake"])
        game.snake_env.last_snake_position = copy.deepcopy(
            game_state["last_snake_position"])
        game.snake_env.direction = game_state["direction"]
        game.snake_env.lost = game_state["lost"]
        game.snake_env.won = game_state["won"]
        game.game_finished = game_state["lost"] or game_state["won"]
        game.snake_env.score = game_state["score"]
        game.snake_env.price_position = copy.deepcopy(game_state["food"])

        version, internal, gauss = game_state["rng_state"]
        game.snake_env.rng.setstate(version, tuple(internal), gauss)
        return game
