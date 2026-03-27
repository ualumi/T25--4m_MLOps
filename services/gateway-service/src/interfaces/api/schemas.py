"""Схемы API для сервиса gateway."""

from pydantic import BaseModel, Field


class ConnectRequest(BaseModel):
    user_id: str = Field(min_length=1)


class ConnectResponse(BaseModel):
    user_id: str
    session_token: str


class PredictRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_token: str = Field(min_length=1)
    features: list[float] = Field(min_length=1)


class PredictResponse(BaseModel):
    score: float
