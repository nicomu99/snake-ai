"""Constants shared by the game, the policy and the GUI.

Attributes:
    TILE_COUNT (int): The number of tiles on each side of the square board.
        ``PolicyModel``'s input layer size depends on it, so saved models
        only load with the ``TILE_COUNT`` they were trained with. It must
        match ``BOARD_SIZE`` in ``frontend/src/config.ts``.
    TILE_SIZE (int): The base tile size for the Tk GUI. ``gui.view`` scales
        it by 10 to get the tile size in pixels.
"""

TILE_COUNT = 8
TILE_SIZE = 10
