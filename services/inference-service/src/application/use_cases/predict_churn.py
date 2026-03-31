"""Сценарий получения прогноза по оттоку."""

from dataclasses import dataclass

from src.application.ports.scoring import ScoringPort
from src.domain.entities.prediction_input import PredictionInput


@dataclass(frozen=True)
class PredictResult:
    score: float


@dataclass(frozen=True)
class BatchPredictResultItem:
    client_id: str
    score: float


@dataclass(frozen=True)
class BatchPredictResult:
    predictions: list[BatchPredictResultItem]


@dataclass(frozen=True)
class BatchPredictClientPayload:
    client_id: str
    features: list[float]


class PredictChurnUseCase:
    def __init__(self, scorer: ScoringPort) -> None:
        self._scorer = scorer

    def execute(self, payload: PredictionInput) -> PredictResult:
        score = self._scorer.score(payload.features)
        return PredictResult(score=score)

    def execute_batch(self, clients: list[BatchPredictClientPayload]) -> BatchPredictResult:
        predictions = [
            BatchPredictResultItem(
                client_id=client.client_id,
                score=float(self._scorer.score(client.features)),
            )
            for client in clients
        ]
        return BatchPredictResult(predictions=predictions)
