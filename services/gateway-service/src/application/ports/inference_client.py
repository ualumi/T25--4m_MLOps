"""Порт для взаимодействия с сервисом инференса."""

from __future__ import annotations

from abc import ABC, abstractmethod


class InferenceClientPort(ABC):
    @abstractmethod
    def predict(self, features: list[float], api_key: str) -> dict[str, float | str]:
        """Отправляет данные в сервис инференса и возвращает результат."""
