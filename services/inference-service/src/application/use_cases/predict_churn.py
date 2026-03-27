"""Сценарий получения прогноза по оттоку."""

from dataclasses import dataclass

from src.application.ports.scoring import ScoringPort
from src.domain.entities.prediction_input import PredictionInput


@dataclass(frozen=True)
class PredictResult:
    score: float


class PredictChurnUseCase:
    def __init__(self, scorer: ScoringPort) -> None:
        self._scorer = scorer

    def execute(self, payload: PredictionInput) -> PredictResult:
        score = self._scorer.score(payload.features)
        return PredictResult(score=score)
