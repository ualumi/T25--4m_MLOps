"""Сценарий построения ранжированного целевого сегмента."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.application.ports.scoring import ScoringPort
from src.domain.entities.scoring_candidate import ScoringCandidate


@dataclass(frozen=True)
class RankedUser:
    user_id: str
    probability_of_inactivity: float
    rank: int


@dataclass(frozen=True)
class TargetSegmentResult:
    top_share: float
    total_users: int
    segment: list[RankedUser]
    top_segment: list[RankedUser]


class BuildTargetSegmentUseCase:
    def __init__(self, scorer: ScoringPort) -> None:
        self._scorer = scorer

    def execute(
        self, candidates: list[ScoringCandidate], top_share: float = 0.2
    ) -> TargetSegmentResult:
        if not candidates:
            raise ValueError("Список кандидатов не может быть пустым.")
        if top_share <= 0 or top_share > 1:
            raise ValueError("top_share должна быть в диапазоне (0, 1].")

        scored = [
            (candidate.user_id, float(self._scorer.score(candidate.features)))
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        ranked = [
            RankedUser(
                user_id=user_id,
                probability_of_inactivity=score,
                rank=index + 1,
            )
            for index, (user_id, score) in enumerate(scored)
        ]

        top_segment_size = max(1, math.ceil(len(ranked) * top_share))
        return TargetSegmentResult(
            top_share=top_share,
            total_users=len(ranked),
            segment=ranked,
            top_segment=ranked[:top_segment_size],
        )
