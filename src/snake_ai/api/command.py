from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class Start(BaseModel):
    type: Literal["start"]


class Pause(BaseModel):
    type: Literal["pause"]


class Resume(BaseModel):
    type: Literal["resume"]


class Reset(BaseModel):
    type: Literal["reset"]
    seed: int


class Speed(BaseModel):
    type: Literal["speed"]
    tick_ms: int


class SetModel(BaseModel):
    type: Literal["set_model"]
    model_id: str


class Command(BaseModel):
    command: Start | Pause | Resume | Reset | Speed | SetModel = (
        Field(discriminator="type")
    )

def parse_message(message: dict):
    try:
        Command(command=message)
    except ValidationError as e:
        print(e)
