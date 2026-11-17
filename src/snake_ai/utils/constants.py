"""Constants shared by the game, the policy and the GUI.

Attributes:
    TILE_COUNT (int): The number of tiles on each side of the square board.
        ``PolicyModel`` is hard-coded for 5, so changing this breaks the
        network and the saved models.
    TILE_SIZE (int): The base tile size for the Tk GUI. ``gui.view`` scales
        it by 10 to get the tile size in pixels.
"""

TILE_COUNT = 5
TILE_SIZE = 30
