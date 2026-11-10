from typing import Literal

from pydantic import BaseModel, Field


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
