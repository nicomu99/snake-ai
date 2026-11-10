from json import JSONDecodeError
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from snake_ai.registries import GameSessionRegistry

from .command_message import parse_message
from .deps import get_session_registry
from .errors import InvalidCommandError
from .web_socket_sink import WebSocketSink

router = APIRouter()


@router.websocket("/ws/play")
async def play(
    ws: WebSocket,
    registry: Annotated[GameSessionRegistry, Depends(get_session_registry)],
    game_id: str | None = None,
):
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
