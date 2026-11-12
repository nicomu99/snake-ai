"""Keeps track of all running games by id."""

# import json
import asyncio
import logging

# import redis.asyncio as redis
from snake_ai.ai import Agent
from snake_ai.api.game_runner import GameRunner
from snake_ai.api.web_socket_sink import WebSocketSink

logger = logging.getLogger(__name__)


class GameRunnerRegistry:
    """Creates, finds and removes ``GameRunner``s.

    Runners are kept in memory under keys of the form
    ``game_runner:<id>``. The commented-out Redis code shows a planned
    persistence backend.

    Args:
        agent: The shared agent that every new game uses.
    """

    def __init__(self, agent: Agent, ):  # r: redis.Redis):
        # self._r = r
        self._agent = agent
        self._runners: dict[str, GameRunner] = {}

    async def attach(self, sink: WebSocketSink, game_runner_id: str | None):
        """Attaches a sink to an existing game, or to a new one.

        A new game is created if ``game_runner_id`` is None or unknown.

        Args:
            sink: The client's sink.
            game_runner_id: The id of the game to reconnect to, if any.

        Returns:
            The runner the sink is now attached to.
        """
        logger.info("Attaching new sink")
        if game_runner_id is None or not self.check_key(game_runner_id):
            runner = GameRunner(self._agent)
            await self.add(runner)
        else:
            runner = await self.get(game_runner_id)
        await runner.attach(sink)
        return runner

    async def detach(self, sink: WebSocketSink, game_runner: GameRunner):
        """Detaches a sink from its runner, see ``GameRunner.detach``.

        Args:
            sink: The sink of the client that disconnected.
            game_runner: The runner the sink was attached to.
        """
        await game_runner.detach(sink)

    def remove_expired(self, now: float):
        """Cancels and removes every runner that has been detached too long.

        Args:
            now: The current ``time.monotonic()`` value.
        """
        logger.debug("Removing expired game sessions")
        for runner_id in list(self._runners.keys()):
            game_runner = self._runners[runner_id]
            if not game_runner.is_expired(now):
                continue
            logger.info("Deleting game session %s", runner_id)
            game_runner.cancel()
            del self._runners[runner_id]

    async def close_all(self):
        """Stops every runner and clears the registry. Used on shutdown."""
        runners = list(self._runners.items())
        results = await asyncio.gather(
            *(runner.aclose() for _, runner in runners),
            return_exceptions=True
        )
        for runner_id, _, result in zip(runners, results, strict=True):
            if result is None:
                continue
            logger.error("Runner %s failed", runner_id, exc_info=result)
        self._runners.clear()

    def check_key(self, game_runner_id: str) -> bool:
        """Returns whether a runner with this id exists."""
        return self._key(game_runner_id) in self._runners

    async def add(self, game_runner: GameRunner) -> str:
        """Stores a runner under its own id.

        Args:
            game_runner: The runner to store.

        Returns:
            The runner's id, 12 hex characters.
        """
        game_runner_id = game_runner.id
        key = self._key(game_runner_id)
        # await self._r.set(
        #     key, json.dumps(game_session.to_dict())
        # )
        self._runners[key] = game_runner
        return game_runner_id

    async def get(self, game_runner_id: str) -> GameRunner:
        """Looks up a runner by id.

        Args:
            game_runner_id: The id returned by ``add``.

        Returns:
            The runner.

        Raises:
            KeyError: If the id is unknown. Check with ``check_key`` first.
        """
        # raw = await self._r.get(self._key(game_session_id))
        return self._runners[self._key(game_runner_id)]

    async def save(self, game_runner_id: str, game_runner: GameRunner) -> None:
        """Saves a runner to Redis. Not implemented yet, does nothing.

        Args:
            game_runner_id: The id of the runner to save.
            game_runner: The runner to save.
        """
        # await self._r.set(
        #     self._key(game_session_id), json.dumps(game_session.to_dict())
        # )

    async def delete(self, game_runner_id: str) -> bool:
        """Removes a runner from Redis. Not implemented yet, returns None.

        Args:
            game_runner_id: The id returned by ``add``.

        Returns:
            Planned: True if it existed and was removed, False if the id was
            unknown.
        """
        # deleted_keys = await self._r.delete(self._key(game_session_id))
        # return bool(deleted_keys)

    @staticmethod
    def _key(game_runner_id: str):
        """Returns the storage key for a runner id."""
        return f"game_runner:{game_runner_id}"
