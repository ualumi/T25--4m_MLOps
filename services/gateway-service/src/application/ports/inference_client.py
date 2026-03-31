"""Порт для взаимодействия с сервисом инференса."""

from __future__ import annotations

from abc import ABC, abstractmethod


class InferenceClientPort(ABC):
    @abstractmethod
    def predict(self, features: list[float], api_key: str) -> dict[str, float]:
        """Отправляет данные в сервис инференса и возвращает результат."""

    @abstractmethod
    def predict_batch(
        self, clients: list[dict[str, str | list[float]]], api_key: str
    ) -> dict[str, list[dict[str, str | float]]]:
        """Отправляет батч данных в сервис инференса и возвращает результаты."""
