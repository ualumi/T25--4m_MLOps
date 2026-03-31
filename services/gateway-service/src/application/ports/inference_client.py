"""Порт для взаимодействия с сервисом инференса."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class BatchPredictionClientPayload(TypedDict):
    client_id: str
    features: list[float]


class BatchPredictionResultItem(TypedDict):
    client_id: str
    score: float


class BatchPredictionResult(TypedDict):
    predictions: list[BatchPredictionResultItem]


class SegmentClientPayload(TypedDict):
    user_id: str
    features: list[float]


class SegmentResultUser(TypedDict):
    user_id: str
    probability_of_inactivity: float
    rank: int


class SegmentResult(TypedDict):
    top_share: float
    total_users: int
    segment: list[SegmentResultUser]
    top_segment_size: int
    top_segment: list[SegmentResultUser]
    result_uri: str | None


class InferenceClientPort(ABC):
    @abstractmethod
    def predict(self, features: list[float], api_key: str) -> dict[str, float]:
        """Отправляет данные в сервис инференса и возвращает результат."""

    @abstractmethod
    def predict_batch(
        self, clients: list[BatchPredictionClientPayload], api_key: str
    ) -> BatchPredictionResult:
        """Отправляет батч данных в сервис инференса и возвращает результаты."""

    @abstractmethod
    def segment(
        self,
        users: list[SegmentClientPayload],
        top_share: float,
        api_key: str,
    ) -> SegmentResult:
        """Строит сегмент пользователей и возвращает ранжированный результат."""
