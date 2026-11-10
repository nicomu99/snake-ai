import asyncio
import logging

from snake_ai.ai import Agent

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
        self.sink = None
        self.running = asyncio.Event()
        self.task = asyncio.create_task(self._loop())
        self.tick_ms = 16

    async def _loop(self):
        logger.info("Starting game session loop")
        while not self.model.lost:
            await self.running.wait()
            await self.agent_step()
            await asyncio.sleep(self.tick_ms / 1000)
        self.pause()

    async def handle(self, command: str):
        _COMMAND_MAP = {
            "pause": self.pause,
            "resume": self.start_or_resume,
            "start": self.start_or_resume,
            "reset": self.reset,
        }
        func = _COMMAND_MAP[command]
        func()

    def start_or_resume(self):
        self.running.set()

    def pause(self):
        self.running.clear()

    def reset(self):
        state = self.model.reset()
        self.task = asyncio.create_task(self._loop())
        return state

    def subscribe(self, listener):
        """Register ``listener(snapshot)``, called after every state change."""
        self._listeners.append(listener)

    async def step(self, action):
        """Perform a move. ``action`` is a direction string or an action int."""
        direction = ACTIONS[action] if isinstance(action, int) else action
        logger.debug("Performing move into direction %s", direction)

        reward, state, done = self.model.perform_move(direction)
        await self._notify()
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

    async def _notify(self):
        snapshot = self.snapshot()
        if self.sink is not None:
            await self.sink.send_frame(snapshot)
        if not self._listeners:
            return
        for listener in self._listeners:
            listener(snapshot)

    def init_message(self):
        pass

    def to_dict(self):
        return self.snapshot()

    def from_dict(self, agent: Agent):
        # todo
        pass
