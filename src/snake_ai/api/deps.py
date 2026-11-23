"""FastAPI dependencies for the API routes."""

from fastapi.requests import HTTPConnection

from snake_ai.ai import PolicySelector
from snake_ai.api.game_runner_registry import GameRunnerRegistry


def get_registry(request: HTTPConnection) -> GameRunnerRegistry:
    """Returns the shared game runner registry.

    Args:
        request: The current HTTP or WebSocket connection.

    Returns:
        The registry stored on ``app.state`` by the ``lifespan`` in ``main.py``.
    """
    return request.app.state.runner_registry


def get_policy_selector(request: HTTPConnection) -> PolicySelector:
    """Returns the shared policy selector.

    Args:
        request: The current HTTP or WebSocket connection.

    Returns:
        The selector stored on ``app.state`` by the ``lifespan`` in ``main.py``.
    """
    return request.app.state.policy_selector
