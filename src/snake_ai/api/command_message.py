from pydantic import BaseModel, Field, ValidationError

from snake_ai.snake import Pause, Reset, Resume, SetModel, Speed, Start

from .errors import InvalidCommandError


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
