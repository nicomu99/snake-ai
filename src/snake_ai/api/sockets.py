from typing import Annotated

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from snake_ai.registries import GameSessionRegistry
from .web_socket_sink import WebSocketSink
from .deps import get_session_registry

router = APIRouter()


def get_command(message: dict):
    print(message)
    if "type" not in message:
        # TODO: raise
        pass
    command = message["type"]
    if command not in ["start", "pause", "resume"]:
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
        registry.detach(sink, session)
