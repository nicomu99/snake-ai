import logging

logger = logging.getLogger(__name__)

# Index = action int produced by the agent
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class GameSession:
    """Headless game core that knows nothing about how the game is
    displayed."""

    def __init__(self, model, agent):
        self.model = model
        self.agent = agent
        self._listeners = []

    def subscribe(self, listener):
        """Register ``listener(snapshot)``, called after every state change."""
        self._listeners.append(listener)

    def reset(self):
        state = self.model.reset()
        self._notify()
        return state

    def step(self, action):
        """Perform a move. ``action`` is a direction string or an action int."""
        direction = ACTIONS[action] if isinstance(action, int) else action
        logger.info("Performing move into direction %s", direction)

        reward, state, done = self.model.perform_move(direction)
        self._notify()
        return reward, state, done

    def agent_step(self):
        """Let the agent choose and perform one greedy move."""
        action = self.agent.act(self.model.get_state())
        return self.step(action)

    def snapshot(self):
        """Plain, serializable view of the current game state."""
        return {
            "snake": [list(pos) for pos in self.model.snake_position],
            "food": list(self.model.price_position),
            "score": self.model.score,
            "lost": self.model.lost,
        }

    def _notify(self):
        if not self._listeners:
            return
        snapshot = self.snapshot()
        for listener in self._listeners:
            listener(snapshot)
