"""The API routes.

``WS /ws/play`` lets a client watch and control a game, ``GET /models`` lists
the models a game can be played with.
"""

from json import JSONDecodeError
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket

from snake_ai.ai import PolicySelector
from snake_ai.api.command_message import parse_message
from snake_ai.api.deps import get_policy_selector, get_registry
from snake_ai.api.errors import InvalidCommandError
from snake_ai.api.game_runner_registry import GameRunnerRegistry
from snake_ai.api.schemas import ModelItem, ModelResponse
from snake_ai.api.web_socket_sink import WebSocketSink

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
    connected to it is disconnected. Every message the server sends has a
    ``type`` field. The first one is always an ``init_message``: the
    current snapshot plus a ``game_id``, which the client can keep to
    reconnect later.

    After that, the server sends a ``snapshot`` message (see
    ``GameRunner.snapshot``) after every move, and an ``end_message`` (see
    ``GameRunner.end_message``) when the game is lost or won, and a
    ``model_changed`` message (see ``GameRunner.model_update_message``)
    after a ``set_model`` command. The client
    sends commands as JSON objects (see ``command_message``). Invalid input
    gets ``{"type": "error", "message": ...}`` back.

    When the client disconnects, the game is paused and detached. The
    sweeper deletes it if nobody reconnects in time. If there are too many
    games, the oldest one is evicted: its client gets a ``game_canceled``
    message and the socket is closed with code 4001.

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
    finally:
        await registry.detach(sink, session)


@router.get("/models")
async def retrieve_models(
    policy_selector: Annotated[PolicySelector, Depends(get_policy_selector)],
) -> ModelResponse:
    """Lists the models that were loaded at startup.

    Models listed in the catalog whose file was missing are left out, so
    every returned id can be passed to ``set_model``.

    Args:
        policy_selector: The shared selector holding the loaded models.

    Returns:
        The default model id and the loaded models, in catalog order.
    """
    default_model_id = policy_selector.default_model_id
    models = [
        ModelItem(id=mid, name=m_entry.name)
        for mid, m_entry in policy_selector.model_catalog.items()
    ]
    return ModelResponse(default_model_id=default_model_id, models=models)
