from fastapi.requests import HTTPConnection

from .game_runner_registry import GameRunnerRegistry


def get_registry(request: HTTPConnection) -> GameRunnerRegistry:
    return request.app.state.runner_registry
