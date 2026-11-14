"""Constants shared by the game, the policy and the GUI.

Attributes:
    TILE_COUNT (int): The number of tiles on each side of the square board.
        ``PolicyModel`` is hard-coded for 5, so changing this breaks the
        network and the saved models.
    TILE_SIZE (int): The base tile size for the Tk GUI. ``gui.view`` scales
        it by 10 to get the tile size in pixels.
    MODEL_BASE_PATH (Path): The directory that trained models are saved to
        and loaded from by default. It is relative to the current working
        directory.
"""

from pathlib import Path

TILE_COUNT = 5
TILE_SIZE = 30

MODEL_BASE_PATH = Path("policy_models")
