"""Runs one game in the background and streams it to a client."""

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
    """Owns one ``Game`` and plays it in an asyncio task.

    The loop waits until the runner is started, then on each tick lets the
    agent make a move, sends the snapshot to the attached sink and sleeps for
    ``tick_ms``. When the game is lost or won, it sends an end message and
    pauses.

    Only one sink can be attached at a time. Attaching a new one closes the
    old one. A runner without a sink pauses and is deleted by the sweeper
    after ``_DELETION_DELAY`` seconds.

    Must be created inside a running event loop, because ``__init__`` starts
    the loop task.

    Args:
        agent: The shared agent that plays the game.

    Attributes:
        id: A random id of 12 hex characters.
        game: The game being played.
        sink: The attached sink, or None while detached.
        running: Set while the game loop should make moves.
        task: The loop task.
        tick_ms: The delay between moves in milliseconds.
        detached_since: The ``time.monotonic()`` time the sink was detached,
            or None while a sink is attached.
    """

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
        """Plays the game while ``running`` is set, until cancelled."""
        logger.info("Starting game session loop")
        while True:
            await self.running.wait()
            self.game.agent_step()
            snapshot = self.game.snapshot()
            await self.send_frame(snapshot)
            if self.game.snake_env.lost or self.game.snake_env.won:
                await self.send_frame(self.end_message())
                self.pause()
            await asyncio.sleep(self.tick_ms / 1000)
    
    async def send_frame(self, message: dict):
        """Sends a message to the sink, if one is attached.

        Errors while sending are logged and not raised.

        Args:
            message: A JSON-serializable dict.
        """
        try:
            if self.sink is None:
                return
            await self.sink.send_frame(message)
        except Exception:
            logger.exception("Sending frame failed")

    async def handle(
        self,
        command: Pause | Reset | Resume | SetModel | Speed | Start,
    ):
        """Applies a command received from the client.

        Args:
            command: The parsed command.
        """
        logger.info("Handling command %s", command)
        match command:
            case Pause():
                self.pause()
            case Reset():
                await self.reset(command.seed)
            case Resume():
                self.start_or_resume()
            case SetModel():
                self.set_model(command.model_id)
            case Speed():
                self.set_speed(command.tick_ms)
            case Start():
                self.start_or_resume()

    def start_or_resume(self):
        """Lets the loop make moves."""
        self.running.set()

    def pause(self):
        """Stops the loop from making moves after the current tick."""
        self.running.clear()

    async def reset(self, _seed: int | None = None):
        """Pauses, starts a new game and sends the init message again.

        Args:
            _seed: The random seed for the new game. Not used yet.
        """
        # TODO: what to do with seed
        self.pause()
        self.game.reset()
        await self.sink.send_frame(self.init_message())

    def set_model(self, model_id: str):
        """Switches the model that plays the game. Not implemented yet."""
        # TODO
        pass

    def set_speed(self, tick_ms: int):
        """Sets the delay between moves, effective from the next tick.

        Args:
            tick_ms: The delay in milliseconds.
        """
        self.tick_ms = tick_ms

    async def attach(self, sink: WebSocketSink):
        """Attaches a sink, closes the previous one and sends the init message.

        Args:
            sink: The new client's sink.
        """
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
        """Detaches a sink and pauses the game.

        Does nothing if ``sink`` was already replaced by a newer one.

        Args:
            sink: The sink of the client that disconnected.
        """
        if sink is not self.sink:
            return
        self.sink = None
        self.detached_since = time.monotonic()
        logger.info("Detaching game session %s", self.id)
        self.pause()

    def is_expired(self, now: float) -> bool:
        """Returns whether the runner has been detached for too long.

        Args:
            now: The current ``time.monotonic()`` value.

        Returns:
            True if detached for more than ``_DELETION_DELAY`` seconds.
        """
        if self.detached_since is None:
            logger.debug("Game session not detached")
            return False
        expired_time = now - self.detached_since
        logger.debug("Game session detached for %.2f seconds", expired_time)
        return expired_time > self._DELETION_DELAY

    def cancel(self):
        """Cancels the loop task without waiting for it to finish."""
        self.task.cancel()

    async def aclose(self):
        """Cancels the loop task and waits for it to finish."""
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            logger.info("Game session %s stopped", self.id)

    def init_message(self) -> dict:
        """Returns the first frame sent to a client: ``{"game_id": ...}``."""
        return {"game_id": self.id}

    def end_message(self):
        """Returns the frame sent when the game ends.

        Returns:
            A dict with ``reason`` (``"collision"`` or ``"won_game"``),
            ``score``, ``step_count``, ``snake_length`` and ``won``.
        """
        snapshot = self.game.snapshot()
        reason = "collision" if snapshot["lost"] else "won_game"
        return {
            "reason": reason,
            "score": snapshot["score"],
            "step_count": snapshot["step_count"],
            "snake_length": len(snapshot["snake"]),
            "won": snapshot["won"],
        }
