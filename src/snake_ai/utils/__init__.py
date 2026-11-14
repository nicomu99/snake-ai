"""Shared helpers used across the Snake AI package.

Exports:
    MODEL_BASE_PATH: The default directory for saved models.
    TILE_COUNT: The number of tiles on each side of the board.
    TILE_SIZE: The base tile size for the Tk GUI.
    configure_logging: Sets up console and rotating file logging.
"""

from .constants import MODEL_BASE_PATH, TILE_COUNT, TILE_SIZE
from .log_config import configure_logging

__all__ = [
    "MODEL_BASE_PATH",
    "TILE_COUNT",
    "TILE_SIZE",
    "configure_logging",
]
