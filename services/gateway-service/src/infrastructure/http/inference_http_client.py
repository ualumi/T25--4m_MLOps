"""HTTP-адаптер для обращения к сервису инференса."""

from __future__ import annotations

import httpx

from src.application.ports.inference_client import (
    BatchPredictionClientPayload,
    BatchPredictionResult,
    InferenceClientPort,
)


class InferenceHttpClient(InferenceClientPort):
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def predict(self, features: list[float], api_key: str) -> dict[str, float]:
        response = httpx.post(
            f"{self._base_url}/predict",
            json={"features": features},
            headers={"x-api-key": api_key},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(
        self, clients: list[BatchPredictionClientPayload], api_key: str
    ) -> BatchPredictionResult:
        response = httpx.post(
            f"{self._base_url}/predict/batch",
            json={"clients": clients},
            headers={"x-api-key": api_key},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()
