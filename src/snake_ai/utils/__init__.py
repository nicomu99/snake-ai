"""Shared helpers used across the Snake AI package.

Exports:
    configure_logging: Sets up console and rotating file logging.
    TileMap: Board size and tile pixel size constants.
"""

from .log_config import configure_logging
from .tile_map import TileMap

__all__ = [
    "configure_logging",
    "TileMap",
]
