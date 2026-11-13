"""Shared helpers used across the Snake AI package.

Exports:
    TILE_COUNT: The number of tiles per row.
    TILE_SIZE: Size of the tiles in the UI.
    configure_logging: Sets up console and rotating file logging.
"""

from .constants import TILE_COUNT, TILE_SIZE
from .log_config import configure_logging

__all__ = [
    "TILE_COUNT",
    "TILE_SIZE",
    "configure_logging",
]
