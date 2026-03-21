"""Адаптер для работы с моделью, сохранённой через joblib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.application.ports.scoring import ScoringPort


class JoblibModelScorer(ScoringPort):
    def __init__(self, model_path: str) -> None:
        self._path = Path(model_path)
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if not self._path.exists():
            raise FileNotFoundError(f"Model file not found: {self._path}")
        return joblib.load(self._path)

    def score(self, features: list[float]) -> float:
        if self._model is None:
            self._model = self._load_model()

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba([features])[0][1]
            return float(proba)

        prediction = self._model.predict([features])[0]
        return float(prediction)
