from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from snake_ai.registries import GameSessionRegistry

from .deps import get_session_registry
from .web_socket_sink import WebSocketSink

router = APIRouter()


def get_command(message: dict):
    print(message)
    if "type" not in message:
        # TODO: raise
        pass
    command = message["type"]
    if command not in [
        "start",
        "pause",
        "resume",
        "reset",
        "speed",
        "set_model",
    ]:
        # TODO: raise
        pass
    return command


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
            command = get_command(await ws.receive_json())
            await session.handle(command)
    except WebSocketDisconnect:
        await registry.detach(sink, session)
