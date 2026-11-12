"""The WebSocket endpoint clients use to watch and control a game."""

from json import JSONDecodeError
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from .command_message import parse_message
from .deps import get_registry
from .errors import InvalidCommandError
from .game_runner_registry import GameRunnerRegistry
from .web_socket_sink import WebSocketSink

router = APIRouter()


@router.websocket("/ws/play")
async def play(
    ws: WebSocket,
    registry: Annotated[GameRunnerRegistry, Depends(get_registry)],
    game_id: str | None = None,
):
    """Connects a client to a game and handles its commands.

    Without a ``game_id``, or with an unknown one, a new game is created.
    With a known id, the client takes over that game, and any other client
    connected to it is disconnected. The first frame sent is always
    ``{"game_id": ...}``, which the client can keep to reconnect later.

    After that, the server sends a snapshot (see ``Game.snapshot``) after
    every move, and an end message when the game is lost or won. The client
    sends commands as JSON objects (see ``command_message``). Invalid input
    gets ``{"type": "error", "message": ...}`` back.

    When the client disconnects, the game is paused and detached. The
    sweeper deletes it if nobody reconnects in time.

    Args:
        ws: The client's WebSocket connection.
        registry: The shared game runner registry.
        game_id: The id of a game to reconnect to, from the query string.
    """
    await ws.accept()
    sink = WebSocketSink(ws)
    session = await registry.attach(sink, game_id)   # resume, or create a new game
    try:
        while True:
            try:
                command_message = parse_message(await ws.receive_json())
                await session.handle(command_message.command)
            except InvalidCommandError as e:
                await ws.send_json({"type": "error", "message": str(e)})
            except JSONDecodeError as _:
                await ws.send_json({
                    "type": "error",
                    "message": "Message could not be decoded. Please send valid JSON."
                })
    except WebSocketDisconnect:
        await registry.detach(sink, session)
