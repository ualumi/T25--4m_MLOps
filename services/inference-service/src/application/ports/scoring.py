"""Порт скоринга, который использует прикладной слой."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ScoringPort(ABC):
    @abstractmethod
    def score(self, features: list[float]) -> float:
        """Возвращает вероятность оттока в диапазоне от 0 до 1."""
