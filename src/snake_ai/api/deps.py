from fastapi.requests import HTTPConnection

from snake_ai.registries import GameSessionRegistry


def get_session_registry(request: HTTPConnection) -> GameSessionRegistry:
    return request.app.state.game_sessions
