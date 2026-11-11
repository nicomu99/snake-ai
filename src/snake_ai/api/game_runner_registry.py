# import json
import asyncio
import logging

# import redis.asyncio as redis
from snake_ai.ai import Agent
from snake_ai.api.game_runner import GameRunner
from snake_ai.api.web_socket_sink import WebSocketSink

logger = logging.getLogger(__name__)


class GameRunnerRegistry:
    def __init__(self, agent: Agent, ):  # r: redis.Redis):
        # self._r = r
        self._agent = agent
        self._runners: dict[str, GameRunner] = {}

    async def attach(self, sink: WebSocketSink, game_runner_id: str | None):
        logger.info("Attaching new sink")
        if game_runner_id is None or not self.check_key(game_runner_id):
            runner = GameRunner(self._agent)
            await self.add(runner)
        else:
            runner = await self.get(game_runner_id)
        await runner.attach(sink)
        return runner

    async def detach(self, sink: WebSocketSink, game_runner: GameRunner):
        await game_runner.detach(sink)

    def remove_expired(self, now: float):
        logger.debug("Removing expired game sessions")
        for runner_id in list(self._runners.keys()):
            game_runner = self._runners[runner_id]
            if not game_runner.is_expired(now):
                continue
            logger.info("Deleting game session %s", runner_id)
            game_runner.cancel()
            del self._runners[runner_id]

    async def close_all(self):
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
        return self._key(game_runner_id) in self._runners

    async def add(self, game_runner: GameRunner) -> str:
        """Stores a game session under a new random id.

        Args:
            game_runner: The game session to store.

        Returns:
            The new id, 12 hex characters.
        """
        game_runner_id = game_runner.id
        key = self._key(game_runner_id)
        # await self._r.set(
        #     key, json.dumps(game_session.to_dict())
        # )
        self._runners[key] = game_runner
        return game_runner_id

    async def get(self, game_runner_id: str) -> GameRunner:
        """Looks up a game session by id.

        Args:
            game_runner_id: The id returned by ``add``.

        Returns:
            The game session, or None if the id is unknown.
        """
        # raw = await self._r.get(self._key(game_session_id))
        return self._runners[self._key(game_runner_id)]

    async def save(self, game_runner_id: str, game_runner: GameRunner) -> None:
        """Store a game session into the Redis store.

        Args:
            game_runner_id: Unique identifier of the game session to store.
            game_runner: GameSession data.
        """
        # await self._r.set(
        #     self._key(game_session_id), json.dumps(game_session.to_dict())
        # )

    async def delete(self, game_runner_id: str) -> bool:
        """Removes a game session.

        Args:
            game_runner_id: The id returned by ``add``.

        Returns:
            True if it existed and was removed, False if the id was unknown.
        """
        # deleted_keys = await self._r.delete(self._key(game_session_id))
        # return bool(deleted_keys)

    @staticmethod
    def _key(game_runner_id: str):
        return f"game_runner:{game_runner_id}"
