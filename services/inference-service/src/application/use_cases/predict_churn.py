"""Сценарий получения прогноза по оттоку."""

from dataclasses import dataclass

from src.application.ports.scoring import ScoringPort
from src.domain.entities.prediction_input import PredictionInput
from src.domain.services.churn_label import to_label


@dataclass(frozen=True)
class PredictResult:
    score: float
    label: str


class PredictChurnUseCase:
    def __init__(self, scorer: ScoringPort, threshold: float) -> None:
        self._scorer = scorer
        self._threshold = threshold

    def execute(self, payload: PredictionInput) -> PredictResult:
        score = self._scorer.score(payload.features)
        return PredictResult(score=score, label=to_label(score, self._threshold))
