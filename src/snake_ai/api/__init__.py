"""The WebSocket server that streams agent-played games to clients.

Exports:
    GameRunnerRegistry: Keeps track of all running games.
"""

from .game_runner_registry import GameRunnerRegistry

__all__ = [
    "GameRunnerRegistry",
]
