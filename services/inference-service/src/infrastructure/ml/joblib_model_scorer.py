"""Адаптер для работы с моделью, сохранённой через joblib."""

from __future__ import annotations

from typing import Any

from src.application.ports.scoring import ScoringPort
from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore


class JoblibModelScorer(ScoringPort):
    def __init__(self, model_uri: str, artifact_store: S3ArtifactStore) -> None:
        self._model_uri = model_uri
        self._artifact_store = artifact_store
        self._model: Any | None = None

    def _load_model(self) -> Any:
        return self._artifact_store.load_joblib(self._model_uri)

    def score(self, features: list[float]) -> float:
        if self._model is None:
            self._model = self._load_model()

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba([features])[0][1]
            return float(proba)

        prediction = self._model.predict([features])[0]
        return float(prediction)
