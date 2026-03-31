"""Схемы API для сервиса gateway."""

from pydantic import BaseModel, Field

FEATURES_COUNT = 28


class ConnectRequest(BaseModel):
    user_id: str = Field(min_length=1)


class ConnectResponse(BaseModel):
    user_id: str
    session_token: str


class PredictRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_token: str = Field(min_length=1)
    features: list[float] = Field(min_length=FEATURES_COUNT, max_length=FEATURES_COUNT)


class PredictResponse(BaseModel):
    score: float
    result_uri: str | None = None


class BatchPredictClient(BaseModel):
    client_id: str = Field(min_length=1)
    features: list[float] = Field(min_length=FEATURES_COUNT, max_length=FEATURES_COUNT)


class BatchPredictRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_token: str = Field(min_length=1)
    clients: list[BatchPredictClient] = Field(min_length=1)


class BatchPredictItemResponse(BaseModel):
    client_id: str
    score: float


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPredictItemResponse]
    dataset_uri: str | None = None
    result_uri: str | None = None


class SegmentCandidate(BaseModel):
    user_id: str = Field(min_length=1)
    features: list[float] = Field(min_length=FEATURES_COUNT, max_length=FEATURES_COUNT)


class SegmentRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_token: str = Field(min_length=1)
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
    result_uri: str | None = None
