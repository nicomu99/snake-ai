"""Runs one game in the background and streams it to a client."""

import asyncio
import logging
import time
import uuid

from snake_ai.ai import PolicySelector
from snake_ai.api.command_message import Pause, Reset, Resume, SetModel, Speed, Start
from snake_ai.api.errors import InvalidCommandError
from snake_ai.api.web_socket_sink import WebSocketSink
from snake_ai.snake import Game

logger = logging.getLogger(__name__)


class GameRunner:
    """Owns one ``Game`` and plays it in an asyncio task.

    The loop waits until the runner is started, then on each tick lets the
    policy pick a move, sends the snapshot to the attached sink and sleeps for
    ``tick_ms``. When the game is lost or won, it sends an end message and
    pauses.

    If a tick raises, the exception is logged and the loop keeps running: the
    runner pauses, is marked as ``crashed`` and sends a crash message.
    ``start`` and ``resume`` then do nothing until the client sends ``reset``.

    Only one sink can be attached at a time. Attaching a new one closes the
    old one. A runner without a sink pauses and is deleted by the sweeper
    after ``_DELETION_DELAY`` seconds.

    Must be created inside a running event loop, because ``__init__`` starts
    the loop task.

    Args:
        policy_selector: Provides the policies, which are shared with the
            other runners and only used for inference. The game starts with
            the default policy, and ``set_model`` switches to another one.

    Attributes:
        id: A random id of 12 hex characters.
        game: The game being played.
        sink: The attached sink, or None while detached.
        running: Set while the game loop should make moves.
        task: The loop task.
        tick_ms: The delay between moves in milliseconds.
        detached_since: The ``time.monotonic()`` time the sink was detached,
            or None while a sink is attached.
        crashed: Whether a tick raised an exception since the last reset. The
            game may be half-updated, so it can't be continued.
    """

    _DELETION_DELAY = 5     # seconds

    def __init__(self, policy_selector: PolicySelector):
        self.id = uuid.uuid4().hex[:12]
        self._policy_selector = policy_selector
        self.game = Game(self._policy_selector.choose_default())
        self.sink = None

        self.tick_ms: int = 16
        self.detached_since: float | None = None
        self.crashed: bool = False
        self.running = asyncio.Event()
        self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        """Plays the game while ``running`` is set, until canceled.

        An exception during a tick doesn't end the loop. It is logged, the
        runner pauses, ``crashed`` is set and the crash message is sent.
        """
        logger.info("Starting game session loop")
        while True:
            await self.running.wait()
            try:
                self.game.agent_step()
                snapshot = self.snapshot()
                await self.send_frame(snapshot)
                if self.game.snake_env.lost or self.game.snake_env.won:
                    await self.send_frame(self.end_message())
                    self.pause()
            except Exception as e:
                logger.exception("Game %s crashed", self.id, exc_info=e)
                self.pause()
                self.crashed = True
                await self.send_frame(self.crash_message())
                continue
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
        except Exception as e:
            logger.exception("Sending frame failed", exc_info=e)

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
                self.resume()
            case SetModel():
                self.set_model(command.model_id)
            case Speed():
                self.set_speed(command.tick_ms)
            case Start():
                self.start()

    def start(self):
        """Starts the game.

        If the game is lost, won or crashed, the function does nothing.
        """
        if self.game.snake_env.lost or self.game.snake_env.won or self.crashed:
            return
        self.running.set()

    def resume(self):
        """Lets the loop make moves.

        If the game crashed, the function does nothing.
        """
        if self.crashed:
            return
        self.running.set()

    def pause(self):
        """Stops the loop from making moves after the current tick."""
        self.running.clear()

    async def reset(self, seed: int | None = None):
        """Pauses, starts a new game and sends the init message again.

        This also clears ``crashed``, so the new game can be started.

        Args:
            seed: The random seed for the new game. If None, the current random number
                generator keeps going without being reseeded. Defaults to None.
        """
        self.pause()
        self.game.reset(seed)
        self.crashed = False
        await self.send_frame(self.init_message())

    def set_model(self, model_id: str):
        """Switches the model that plays the game and pauses it.

        The game itself is kept, so the new model continues from the current
        state. The client is not told about the pause yet.

        Args:
            model_id: The id of the new model, as listed in the catalog.

        Raises:
            InvalidCommandError: If no model with this id is available. The
                game is then left unchanged.
        """
        try:
            self.game.policy = self._policy_selector.select_policy(model_id)
            self.pause()
        except ValueError as e:
            logger.warning(
                "Invalid model id %s chosen for game %s",
                model_id, self.id)
            raise InvalidCommandError(
                f"Invalid model id {model_id}. Must be one of "
                f"{list(self._policy_selector.policies.keys())}") from e
        # TODO: send confirmation message
        # TODO: restart?
        pass

    def set_speed(self, tick_ms: int):
        """Sets the delay between moves, effective from the next tick.

        Args:
            tick_ms: The delay in milliseconds.
        """
        self.tick_ms = tick_ms

    async def attach(self, sink: WebSocketSink):
        """Attaches a sink, closes the previous one and sends the init message.

        If the game has crashed, the crash message is sent after the init
        message, so a reconnecting client knows it has to reset.

        Args:
            sink: The new client's sink.
        """
        self.detached_since = None
        old_sink = self.sink
        self.sink = sink
        if old_sink is not None:
            try:
                await old_sink.close()
            except RuntimeError as e:
                logger.exception(e)

        await self.sink.send_frame(self.init_message())
        if self.crashed:
            await self.sink.send_frame(self.crash_message())

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

    def snapshot(self) -> dict:
        """Returns the frame sent to the client after every move.

        Returns:
            The game snapshot (see ``Game.snapshot``) plus
            ``"type": "snapshot"``.
        """
        snapshot = self.game.snapshot()
        return snapshot | {"type": "snapshot"}

    def init_message(self) -> dict:
        """Returns the first frame sent to a client on attach and reset.

        Returns:
            The game snapshot (see ``Game.snapshot``) plus
            ``"type": "init_message"`` and the runner's ``game_id``.
        """
        snapshot = self.game.snapshot()
        return snapshot | {"type": "init_message", "game_id": self.id}

    def end_message(self):
        """Returns the frame sent when the game ends.

        Returns:
            A dict with ``"type": "end_message"``, ``reason`` (the game
            state, i.e. ``"game_won"``, ``"collision"`` or
            ``"direction_reversal"``, see ``SnakeEnv.get_game_state``),
            ``score``, ``step_count``, ``snake_length`` and ``won``.
        """
        snapshot = self.game.snapshot()
        return {
            "type": "end_message",
            "reason": snapshot["game_state"],
            "score": snapshot["score"],
            "step_count": snapshot["step_count"],
            "snake_length": len(snapshot["snake"]),
            "won": snapshot["won"],
        }

    @staticmethod
    def crash_message() -> dict:
        """Returns the frame sent when a tick raised an exception.

        It is sent when the crash happens and again on every attach until the
        game is reset. The exception itself is only logged, not sent.

        Returns:
            A dict with ``"type": "game_crashed"`` and a ``message`` that
            tells the user to reset the game.
        """
        return {
            "type": "game_crashed",
            "message": "Please reset game and try again.",
        }
