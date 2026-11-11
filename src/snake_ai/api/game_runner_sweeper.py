import asyncio
import logging
import time

from snake_ai.api.game_runner_registry import GameRunnerRegistry

logger = logging.getLogger(__name__)


class GameRunnerSweeper:
    def __init__(self, registry: GameRunnerRegistry, delay: float = 5):
        logger.debug("Creating GameSessionSweeper")
        self.task = None
        self.delay = delay
        self.registry = registry

    async def _loop(self):
        logger.info("Starting GameRunnerSweeper with delay %.2f", self.delay)
        while True:
            await asyncio.sleep(self.delay)
            try:
                self.registry.remove_expired(time.monotonic())
            except Exception:
                logger.exception("Sweeper pass failed")

    def start(self):
        if self.task is not None:
            return
        self.task = asyncio.create_task(self._loop())

    async def stop(self):
        if self.task is None:
            return
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            logger.info("Stopped sweeping game sessions")
