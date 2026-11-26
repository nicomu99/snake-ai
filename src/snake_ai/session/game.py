"""A headless Snake game that an agent can play."""

import copy
import logging

from snake_ai.ai.policy import Policy
from snake_ai.snake.snake_env import SnakeEnv

logger = logging.getLogger(__name__)


class Game:
    """A headless game core that knows nothing about how it is displayed.

    ``Game`` wraps a ``SnakeEnv``, lets a ``Policy`` choose moves and
    notifies subscribed listeners with a serializable snapshot after every
    move and reset. Frontends (the WebSocket server, the Tk GUI) build on
    top of it. Training doesn't use ``Game``; ``DQNTrainer`` works on a
    ``SnakeEnv`` directly.

    Args:
        policy: The policy that picks moves in ``agent_step``. It can be
            shared between games.

    Attributes:
        snake_env: The underlying game state and rules.
        policy: The policy used by ``agent_step``.
    """

    def __init__(self, policy: Policy):
        self.snake_env = SnakeEnv()
        self.policy = policy
        self._listeners = []

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
        state = self.snake_env.reset(seed)
        self._notify()
        return state

    def step(self, action: int | str):
        """Performs one move and notifies the listeners.

        Once the game is lost or won, further moves leave the board as it
        is, return a reward of 0 and don't increase ``step_count``. The
        listeners are still notified.

        Args:
            action: A direction string (``"UP"``, ``"DOWN"``, ``"LEFT"``,
                ``"RIGHT"``) or an index into ``snake.snake_env.ACTIONS``.

        Returns:
            A ``(reward, state, lost, won)`` tuple, see
            ``SnakeEnv.perform_move``.

        Raises:
            ValueError: If ``action`` is not a valid direction or index.
        """
        reward, state, lost, won = self.snake_env.perform_move(action)
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
            "step_count": self.snake_env.step_count,
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
        dict and the restored game don't share lists. ``SnakeEnv.game_finished``
        is derived from ``lost`` and ``won``, and the ``game_state`` key is
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
        game.snake_env.step_count = game_state["step_count"]
        game.snake_env.snake_position = copy.deepcopy(game_state["snake"])
        game.snake_env.last_snake_position = copy.deepcopy(
            game_state["last_snake_position"])
        game.snake_env.direction = game_state["direction"]
        game.snake_env.lost = game_state["lost"]
        game.snake_env.won = game_state["won"]
        game.snake_env.game_finished = game_state["lost"] or game_state["won"]
        game.snake_env.score = game_state["score"]
        game.snake_env.price_position = copy.deepcopy(game_state["food"])

        version, internal, gauss = game_state["rng_state"]
        game.snake_env.rng.setstate(version, tuple(internal), gauss)
        return game
