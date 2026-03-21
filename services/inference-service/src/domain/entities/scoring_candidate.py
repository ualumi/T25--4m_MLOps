"""Сущность для пользователя в батчевом скоринге."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringCandidate:
    user_id: str
    features: list[float]
