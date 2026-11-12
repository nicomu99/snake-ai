"""A background task that removes games nobody is connected to."""

import asyncio
import logging
import time

from snake_ai.api.game_runner_registry import GameRunnerRegistry

logger = logging.getLogger(__name__)


class GameRunnerSweeper:
    """Periodically removes expired runners from a registry.

    Every ``delay`` seconds it calls ``GameRunnerRegistry.remove_expired``.
    Errors in one pass are logged and don't stop the sweeper.

    Args:
        registry: The registry to clean up.
        delay: The number of seconds between passes.
    """

    def __init__(self, registry: GameRunnerRegistry, delay: float = 5):
        logger.debug("Creating GameSessionSweeper")
        self.task = None
        self.delay = delay
        self.registry = registry

    async def _loop(self):
        """Runs a sweep every ``delay`` seconds until cancelled."""
        logger.info("Starting GameRunnerSweeper with delay %.2f", self.delay)
        while True:
            await asyncio.sleep(self.delay)
            try:
                self.registry.remove_expired(time.monotonic())
            except Exception:
                logger.exception("Sweeper pass failed")

    def start(self):
        """Starts the sweeper task. Does nothing if it is already running.

        Must be called from inside a running event loop.
        """
        if self.task is not None:
            return
        self.task = asyncio.create_task(self._loop())

    async def stop(self):
        """Cancels the sweeper task and waits for it to finish."""
        if self.task is None:
            return
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            logger.info("Stopped sweeping game sessions")
