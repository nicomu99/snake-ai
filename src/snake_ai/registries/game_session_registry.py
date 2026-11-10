import json
import logging
import uuid

import redis.asyncio as redis

from snake_ai.ai import Agent
from snake_ai.api.web_socket_sink import WebSocketSink
from snake_ai.snake import GameSession, Model

logger = logging.getLogger(__name__)


class GameSessionRegistry:
    def __init__(self, r: redis.Redis, agent: Agent):
        self._r = r
        self._agent = agent
        self._sessions: dict[str, GameSession] = {}

    async def attach(self, sink: WebSocketSink, game_session_id: str | None):
        logger.info("Attaching new sink")
        if game_session_id is None or not self.check_key(game_session_id):
            session = GameSession(Model(), self._agent)
            await self.add(session)
        else:
            session = await self.get(game_session_id)
        old_sink = session.sink
        session.sink = sink
        if old_sink is not None:
            try:
                await old_sink.close()
            except RuntimeError:
                # TODO
                pass

        await sink.send_frame(session.init_message())
        return session

    async def detach(self, sink: WebSocketSink, game_session: GameSession):
        game_session.sink = None
        game_session.pause()
        try:
            await sink.close()
        except RuntimeError:
            # TODO
            pass

    def check_key(self, game_session_id: str) -> bool:
        return game_session_id in self._r

    async def add(self, game_session: GameSession) -> str:
        """Stores a game session under a new random id.

        Args:
            game_session: The game session to store.

        Returns:
            The new id, 12 hex characters.
        """
        game_session_id = uuid.uuid4().hex[:12]
        key = self._key(game_session_id)
        await self._r.set(
            key, json.dumps(game_session.to_dict())
        )
        self._sessions[key] = game_session
        return game_session_id

    async def get(self, game_session_id: str) -> GameSession:
        """Looks up a game session by id.

        Args:
            game_session_id: The id returned by ``add``.

        Returns:
            The game session, or None if the id is unknown.
        """
        # raw = await self._r.get(self._key(game_session_id))
        return self._sessions[self._key(game_session_id)]

    async def save(self, game_session_id: str, game_session: GameSession) -> None:
        """Store a game session into the Redis store.

        Args:
            game_session_id: Unique identifier of the game session to store.
            game_session: GameSession data.
        """
        await self._r.set(
            self._key(game_session_id), json.dumps(game_session.to_dict())
        )

    async def delete(self, game_session_id: str) -> bool:
        """Removes a game session.

        Args:
            game_session_id: The id returned by ``add``.

        Returns:
            True if it existed and was removed, False if the id was unknown.
        """
        deleted_keys = await self._r.delete(self._key(game_session_id))
        return bool(deleted_keys)

    @staticmethod
    def _key(gs_id: str):
        return f"game_session:{gs_id}"
