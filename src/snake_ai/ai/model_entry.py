"""One entry of the model catalog (``models.json``)."""

from pydantic import BaseModel


class ModelEntry(BaseModel):
    """Describes one model that clients can choose.

    Attributes:
        id: The unique id clients use to select the model, e.g. in the
            ``set_model`` command.
        name: A human-readable name, e.g. for a selection list in the UI.
        uri: The model file, relative to the model directory
            (``Settings.model_path``). An absolute path is used as is.
        tile_count: The board size the model was trained for.
        in_channels: The number of input channels the model expects (see
            ``Policy.encode_state``).
    """

    id: str
    name: str
    uri: str
    tile_count: int
    in_channels: int
