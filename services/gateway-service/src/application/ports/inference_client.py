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


class InferenceClientPort(ABC):
    @abstractmethod
    def predict(self, features: list[float], api_key: str) -> dict[str, float]:
        """Отправляет данные в сервис инференса и возвращает результат."""

    @abstractmethod
    def predict_batch(
        self, clients: list[BatchPredictionClientPayload], api_key: str
    ) -> BatchPredictionResult:
        """Отправляет батч данных в сервис инференса и возвращает результаты."""
