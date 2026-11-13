"""A headless Snake game that an agent can play.

Attributes:
    ACTIONS (tuple[str, ...]): Maps the agent's action indices to directions.
        The order must match the outputs of ``PolicyModel``.
"""

import logging

from snake_ai.ai import Agent

from .snake_env import SnakeEnv

logger = logging.getLogger(__name__)

# Index = action int produced by the agent
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class Game:
    """A headless game core that knows nothing about how it is displayed.

    ``Game`` wraps a ``SnakeEnv``, lets an agent choose moves, counts steps
    and notifies subscribed listeners with a serializable snapshot after
    every move. Frontends (the WebSocket server, the Tk GUI) build on top of
    it.

    ``Game`` also provides ``reset()`` and ``step(action)``, so it can be
    passed to ``Agent.train`` as the training environment.

    Args:
        agent: The agent that picks moves in ``agent_step``. It can be shared
            between games.

    Attributes:
        snake_env: The underlying game state and rules.
        agent: The agent used by ``agent_step``.
        step_count: The number of moves since the last reset.
    """

    def __init__(self, agent: Agent):
        self.game_finished = False
        self.snake_env = SnakeEnv()
        self.agent = agent
        self._listeners = []
        self.step_count = 0

    def subscribe(self, listener):
        """Registers a listener that is called after every move.

        Listeners are not called on ``reset``.

        Args:
            listener: A callable that takes the dict returned by ``snapshot``.
        """
        self._listeners.append(listener)

    def reset(self):
        """Starts a new game and resets the step counter.

        Returns:
            The plain state of the new game, see ``SnakeEnv.get_state``.
        """
        self.game_finished = False
        self.step_count = 0
        state = self.snake_env.reset()
        self._notify()
        return state

    def step(self, action):
        """Performs one move and notifies the listeners.

        Args:
            action: A direction string (``"UP"``, ``"DOWN"``, ``"LEFT"``,
                ``"RIGHT"``) or an index into ``ACTIONS``.

        Returns:
            A ``(reward, state, lost, won)`` tuple, see
            ``SnakeEnv.perform_move``.
        """
        if isinstance(action, int):
            if action < 0 or action > 4:
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
        """Lets the agent choose one greedy move and performs it.

        Returns:
            A ``(reward, state, lost, won)`` tuple, see
            ``SnakeEnv.perform_move``.
        """
        action = self.agent.act(self.snake_env.get_state())
        return self.step(action)

    def snapshot(self):
        """Returns a plain, JSON-serializable view of the current game.

        Returns:
            A dict with ``snake`` (a list of ``[x, y]`` positions, head
            first), ``food`` (``[x, y]``), ``score``, ``lost``, ``won`` and
            ``step_count``.
        """
        return {
            "snake": [list(pos) for pos in self.snake_env.snake_position],
            "food": list(self.snake_env.price_position),
            "score": self.snake_env.score,
            "lost": self.snake_env.lost,
            "won": self.snake_env.won,
            "step_count": self.step_count,
        }

    def _notify(self):
        """Calls every listener with a fresh snapshot."""
        snapshot = self.snapshot()
        if not self._listeners:
            return
        for listener in self._listeners:
            listener(snapshot)

    def to_dict(self):
        """Returns the game as a dict for persistence (same as ``snapshot``)."""
        return self.snapshot()

    def from_dict(self, agent: Agent):
        """Restores a game from ``to_dict`` output. Not implemented yet."""
        # todo
        pass
