"""Сценарий запроса предсказания у сервиса инференса."""

from src.application.ports.inference_client import (
    BatchPredictionClientPayload,
    BatchPredictionResult,
    InferenceClientPort,
    SegmentClientPayload,
    SegmentResult,
)
from src.application.ports.session_store import SessionStorePort


class RequestPredictionUseCase:
    def __init__(
        self, store: SessionStorePort, inference: InferenceClientPort, api_key: str
    ) -> None:
        self._store = store
        self._inference = inference
        self._api_key = api_key

    def execute(
        self, user_id: str, token: str, features: list[float]
    ) -> dict[str, float]:
        if not self._store.is_valid(user_id=user_id, token=token):
            raise PermissionError("Invalid session. Connect first.")

        return self._inference.predict(features=features, api_key=self._api_key)

    def execute_batch(
        self,
        user_id: str,
        token: str,
        clients: list[BatchPredictionClientPayload],
    ) -> BatchPredictionResult:
        if not self._store.is_valid(user_id=user_id, token=token):
            raise PermissionError("Invalid session. Connect first.")

        return self._inference.predict_batch(clients=clients, api_key=self._api_key)

    def execute_segment(
        self,
        user_id: str,
        token: str,
        users: list[SegmentClientPayload],
        top_share: float,
    ) -> SegmentResult:
        if not self._store.is_valid(user_id=user_id, token=token):
            raise PermissionError("Invalid session. Connect first.")

        return self._inference.segment(
            users=users,
            top_share=top_share,
            api_key=self._api_key,
        )
