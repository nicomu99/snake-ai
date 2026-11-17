"""The WebSocket server that streams games played by the policy to clients.

Exports:
    GameRunnerRegistry: Keeps track of all running games.
"""

from .game_runner_registry import GameRunnerRegistry

__all__ = [
    "GameRunnerRegistry",
]
