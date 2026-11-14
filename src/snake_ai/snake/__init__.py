"""Snake game rules and the headless game wrapper.

Exports:
    Game: Wraps a ``SnakeEnv`` and a ``Policy``, and notifies listeners.
    SnakeEnv: The raw game state and rules.
"""

from .game import Game
from .snake_env import SnakeEnv

__all__ = [
    "Game",
    "SnakeEnv",
]
