from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .errors import InvalidCommandError


class Start(BaseModel):
    type: Literal["start"]


class Pause(BaseModel):
    type: Literal["pause"]


class Resume(BaseModel):
    type: Literal["resume"]


class Reset(BaseModel):
    type: Literal["reset"]
    seed: int | None = None


class Speed(BaseModel):
    type: Literal["speed"]
    tick_ms: int = Field(ge=1, le=1000)


class SetModel(BaseModel):
    type: Literal["set_model"]
    model_id: str


class CommandMessage(BaseModel):
    command: Start | Pause | Resume | Reset | Speed | SetModel = (
        Field(discriminator="type")
    )


def parse_message(message: dict) -> CommandMessage:
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
