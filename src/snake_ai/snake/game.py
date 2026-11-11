import logging

from snake_ai.ai import Agent

from .snake_env import SnakeEnv

logger = logging.getLogger(__name__)

# Index = action int produced by the agent
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class Game:
    """Headless game core that knows nothing about how the game is
    displayed."""

    def __init__(self, agent: Agent):
        self.snake_env = SnakeEnv()
        self.agent = agent
        self._listeners = []
        self.step_count = 0

    def subscribe(self, listener):
        """Register ``listener(snapshot)``, called after every state change."""
        self._listeners.append(listener)

    def reset(self):
        self.step_count = 0
        state = self.snake_env.reset()
        return state

    def step(self, action):
        """Perform a move. ``action`` is a direction string or an action int."""
        direction = ACTIONS[action] if isinstance(action, int) else action
        logger.debug("Performing move into direction %s", direction)

        reward, state, lost, won = self.snake_env.perform_move(direction)
        self.step_count += 1
        self._notify()
        return reward, state, lost, won

    def agent_step(self):
        """Let the agent choose and perform one greedy move."""
        action = self.agent.act(self.snake_env.get_state())
        return self.step(action)

    def snapshot(self):
        """Plain, serializable view of the current game state."""
        return {
            "snake": [list(pos) for pos in self.snake_env.snake_position],
            "food": list(self.snake_env.price_position),
            "score": self.snake_env.score,
            "lost": self.snake_env.lost,
            "won": self.snake_env.won,
            "step_count": self.step_count,
        }

    def _notify(self):
        snapshot = self.snapshot()
        if not self._listeners:
            return
        for listener in self._listeners:
            listener(snapshot)

    def to_dict(self):
        return self.snapshot()

    def from_dict(self, agent: Agent):
        # todo
        pass
