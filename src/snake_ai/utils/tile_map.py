"""Board dimensions shared by the game, the agent and the GUI."""

from enum import Enum


class TileMap(Enum):
    """Constants that describe the size of the game board.

    Read the numbers through ``.value``, for example ``TileMap.TILE_COUNT.value``.

    Attributes:
        TILE_COUNT: The number of tiles along each side of the square board.
            The policy network's input size is hard-coded for a 5x5 board, so
            changing this breaks the model and the saved ``.pt`` files.
        TILE_SIZE: The base pixel size of one tile. The Tk GUI multiplies it
            by 10 when drawing.
    """

    TILE_COUNT = 5
    TILE_SIZE = 30
