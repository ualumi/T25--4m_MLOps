"""Единый интерфейс конфигурации для всего проекта."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float.") from exc


@dataclass(frozen=True)
class GatewayConfig:
    host: str
    port: int
    inference_service_url: str
    service_api_key: str
    database_url: str


@dataclass(frozen=True)
class InferenceConfig:
    host: str
    port: int
    model_uri: str
    threshold: float
    s3_endpoint_url: str | None
    s3_access_key_id: str | None
    s3_secret_access_key: str | None
    s3_region: str


def get_gateway_config() -> GatewayConfig:
    return GatewayConfig(
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=_int_env("GATEWAY_PORT", 8000),
        inference_service_url=os.getenv(
            "INFERENCE_SERVICE_URL", "http://inference-service:8001"
        ),
        service_api_key=os.getenv("SERVICE_API_KEY", "dev-secret"),
        database_url=os.getenv(
            "GATEWAY_DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/gateway_db",
        ),
    )


def get_inference_config() -> InferenceConfig:
    return InferenceConfig(
        host=os.getenv("INFERENCE_HOST", "0.0.0.0"),
        port=_int_env("INFERENCE_PORT", 8001),
        model_uri=os.getenv("MODEL_URI", "s3://ml-artifacts/models/lgb_model.joblib"),
        threshold=_float_env("CHURN_THRESHOLD", 0.5),
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        s3_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        s3_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
