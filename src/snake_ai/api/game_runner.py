import asyncio
import logging
import time
import uuid

from snake_ai.ai import Agent
from snake_ai.snake import Game

from .command_message import Pause, Reset, Resume, SetModel, Speed, Start
from .web_socket_sink import WebSocketSink

logger = logging.getLogger(__name__)


class GameRunner:

    _DELETION_DELAY = 5     # seconds

    def __init__(self, agent: Agent):
        self.id = uuid.uuid4().hex[:12]
        self._agent = agent
        self.game = Game(self._agent)
        self.sink = None

        self.running = asyncio.Event()
        self.task = asyncio.create_task(self._loop())
        self.tick_ms: int = 16
        self.detached_since: float | None = None

    async def _loop(self):
        logger.info("Starting game session loop")
        while not self.game.snake_env.lost:
            await self.running.wait()
            self.game.agent_step()
            snapshot = self.game.snapshot()
            await self.sink.send_frame(snapshot)
            await asyncio.sleep(self.tick_ms / 1000)
        self.pause()

    async def handle(
        self,
        command: Pause | Reset | Resume | SetModel | Speed | Start,
    ):
        logger.info("Handling command %s", command)
        match command:
            case Pause():
                self.pause()
            case Reset():
                self.reset(command.seed)
            case Resume():
                self.start_or_resume()
            case SetModel():
                self.set_model(command.model_id)
            case Speed():
                self.set_speed(command.tick_ms)
            case Start():
                self.start_or_resume()

    def start_or_resume(self):
        self.running.set()

    def pause(self):
        self.running.clear()

    def reset(self, _seed: int | None = None):
        # TODO
        state = self.game.reset()
        self.task.cancel()
        self.task = asyncio.create_task(self._loop())
        return state

    def set_model(self, model_id: str):
        # TODO
        pass

    def set_speed(self, tick_ms: int):
        self.tick_ms = tick_ms

    async def attach(self, sink: WebSocketSink):
        self.detached_since = None
        old_sink = self.sink
        self.sink = sink
        if old_sink is not None:
            try:
                await old_sink.close()
            except RuntimeError:
                # TODO
                pass

        await self.sink.send_frame(self.init_message())

    async def detach(self, sink: WebSocketSink):
        if sink is not self.sink:
            return
        self.sink = None
        self.detached_since = time.monotonic()
        logger.info("Detaching game session %s", self.id)
        self.pause()

    def is_expired(self, now: float) -> bool:
        if self.detached_since is None:
            logger.debug("Game session not detached")
            return False
        expired_time = now - self.detached_since
        logger.debug("Game session detached for %.2f seconds", expired_time)
        return expired_time > self._DELETION_DELAY

    def cancel(self):
        self.task.cancel()

    async def aclose(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            logger.info("Game session %s stopped", self.id)

    def init_message(self) -> dict:
        return {"hello": "world"}
