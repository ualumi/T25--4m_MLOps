"""Схемы API для сервиса инференса."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1)


class PredictResponse(BaseModel):
    score: float
    label: str


class SegmentCandidate(BaseModel):
    user_id: str = Field(min_length=1)
    features: list[float] = Field(min_length=1)


class SegmentRequest(BaseModel):
    users: list[SegmentCandidate] = Field(min_length=1)
    top_share: float = Field(default=0.2, gt=0, le=1)


class SegmentUserResponse(BaseModel):
    user_id: str
    probability_of_inactivity: float
    rank: int


class SegmentResponse(BaseModel):
    top_share: float
    total_users: int
    segment_size: int
    segment: list[SegmentUserResponse]
