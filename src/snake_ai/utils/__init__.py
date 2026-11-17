"""Shared helpers used across the Snake AI package.

Exports:
    TILE_COUNT: The number of tiles on each side of the board.
    TILE_SIZE: The base tile size for the Tk GUI.
    configure_logging: Sets up console and rotating file logging.
    get_settings: Returns the cached settings read from the environment.
"""

from .constants import TILE_COUNT, TILE_SIZE
from .log_config import configure_logging
from .settings import get_settings

__all__ = [
    "TILE_COUNT",
    "TILE_SIZE",
    "configure_logging",
    "get_settings",
]
