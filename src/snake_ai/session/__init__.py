"""Game sessions that combine the game rules with a policy.

Exports:
    Game: Wraps a ``SnakeEnv`` and a ``Policy``, and notifies listeners.
"""

from .game import Game

__all__ = [
    "Game",
]
