"""Схемы API для сервиса инференса."""

from pydantic import BaseModel, Field

FEATURES_COUNT = 28


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=FEATURES_COUNT, max_length=FEATURES_COUNT)


class PredictResponse(BaseModel):
    score: float


class BatchPredictClient(BaseModel):
    client_id: str = Field(min_length=1)
    features: list[float] = Field(min_length=FEATURES_COUNT, max_length=FEATURES_COUNT)


class BatchPredictRequest(BaseModel):
    clients: list[BatchPredictClient] = Field(min_length=1)


class BatchPredictItemResponse(BaseModel):
    client_id: str
    score: float


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPredictItemResponse]


class SegmentCandidate(BaseModel):
    user_id: str = Field(min_length=1)
    features: list[float] = Field(min_length=FEATURES_COUNT, max_length=FEATURES_COUNT)


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
    segment: list[SegmentUserResponse]
    top_segment_size: int
    top_segment: list[SegmentUserResponse]
