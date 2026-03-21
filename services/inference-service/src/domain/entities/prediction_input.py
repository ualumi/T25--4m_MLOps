"""Сущность со входными данными для инференса."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionInput:
    features: list[float]
