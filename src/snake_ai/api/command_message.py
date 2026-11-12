"""The commands a client can send over the WebSocket, and their parser.

Every command is a JSON object with a ``type`` field that selects the
command, for example ``{"type": "speed", "tick_ms": 50}``.
"""

from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .errors import InvalidCommandError


class Start(BaseModel):
    """Starts the game loop. Behaves the same as ``Resume``."""

    type: Literal["start"]


class Pause(BaseModel):
    """Pauses the game loop."""

    type: Literal["pause"]


class Resume(BaseModel):
    """Resumes a paused game loop. Behaves the same as ``Start``."""

    type: Literal["resume"]


class Reset(BaseModel):
    """Pauses the game and starts a new one.

    Attributes:
        seed: The random seed for the new game. Not used yet.
    """

    type: Literal["reset"]
    seed: int | None = None


class Speed(BaseModel):
    """Changes how long the game loop waits between moves.

    Attributes:
        tick_ms: The delay between moves in milliseconds, 1 to 1000.
    """

    type: Literal["speed"]
    tick_ms: int = Field(ge=1, le=1000)


class SetModel(BaseModel):
    """Switches the model that plays the game. Not implemented yet.

    Attributes:
        model_id: The id of the model to use.
    """

    type: Literal["set_model"]
    model_id: str


class CommandMessage(BaseModel):
    """A wrapper that picks the command class based on the ``type`` field."""

    command: Start | Pause | Resume | Reset | Speed | SetModel = (
        Field(discriminator="type")
    )


def parse_message(message: dict) -> CommandMessage:
    """Validates a decoded JSON message and turns it into a command.

    Args:
        message: The JSON object received from the client.

    Returns:
        The validated message. The command itself is in its ``command``
        field.

    Raises:
        InvalidCommandError: If the message is not a valid command. The
            error message describes the last validation error and, for
            parameter errors, names the parameter.
    """
    try:
        command = CommandMessage.model_validate({"command": message})
        return command
    except ValidationError as e:
        error_dict = e.errors()[-1]
        message = error_dict["msg"]
        error_type = error_dict["type"]
        loc = error_dict["loc"][-1]
        if error_type in [
            "greater_than_equal", "less_than_equal",
            "int_parsing", "string_type", "missing"
        ]:
            message = f"Parameter '{loc}': " + message
        raise InvalidCommandError(message) from e
