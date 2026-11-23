"""Response models of the HTTP routes."""

from pydantic import BaseModel


class ModelItem(BaseModel):
    """One model a game can be played with.

    Attributes:
        id: The model id, as used by the ``set_model`` command.
        name: A human-readable name for display.
    """

    id: str
    name: str


class ModelResponse(BaseModel):
    """The response of ``GET /models``.

    Attributes:
        default_model_id: The id of the model that new games start with.
        models: The loaded models, in catalog order.
    """

    default_model_id: str
    models: list[ModelItem]
