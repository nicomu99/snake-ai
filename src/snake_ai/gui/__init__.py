"""A local Tkinter frontend for the Snake game.

``View`` draws the board and collects user input, and ``Controller`` connects
it to a ``Game``. This is the legacy alternative to the WebSocket server.
"""

from .controller import Controller
from .view import View

__all__ = [
    "Controller",
    "View",
]
