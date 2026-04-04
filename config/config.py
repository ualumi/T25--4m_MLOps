"""Единый интерфейс конфигурации для всего проекта."""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.s3_layout import (
    DEFAULT_DATASET_UPLOAD_PREFIX,
    DEFAULT_MODEL_URI,
    DEFAULT_PREDICTION_RESULTS_PREFIX,
    DEFAULT_RETRAIN_SOURCE_PREFIX,
    DEFAULT_SEGMENT_RESULTS_PREFIX,
)


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
# pylint: disable=too-many-instance-attributes
class GatewayConfig:
    host: str
    port: int
    inference_service_url: str
    service_api_key: str
    database_url: str
    s3_endpoint_url: str | None
    s3_access_key_id: str | None
    s3_secret_access_key: str | None
    s3_region: str
    dataset_upload_bucket: str
    dataset_upload_prefix: str
    retrain_source_bucket: str
    retrain_source_prefix: str
    prediction_results_bucket: str
    prediction_results_prefix: str
    segment_results_bucket: str
    segment_results_prefix: str


@dataclass(frozen=True)
class InferenceConfig:
    host: str
    port: int
    model_uri: str
    s3_endpoint_url: str | None
    s3_access_key_id: str | None
    s3_secret_access_key: str | None
    s3_region: str
    segment_results_bucket: str
    segment_results_prefix: str


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
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        s3_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        s3_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        dataset_upload_bucket=os.getenv("DATASET_UPLOADS_BUCKET", "ml-artifacts"),
        dataset_upload_prefix=os.getenv(
            "DATASET_UPLOADS_PREFIX", DEFAULT_DATASET_UPLOAD_PREFIX
        ),
        retrain_source_bucket=os.getenv(
            "RETRAIN_SOURCE_DATASETS_BUCKET", "ml-artifacts"
        ),
        retrain_source_prefix=os.getenv(
            "RETRAIN_SOURCE_DATASETS_PREFIX", DEFAULT_RETRAIN_SOURCE_PREFIX
        ),
        prediction_results_bucket=os.getenv(
            "PREDICTION_RESULTS_BUCKET", "ml-artifacts"
        ),
        prediction_results_prefix=os.getenv(
            "PREDICTION_RESULTS_PREFIX", DEFAULT_PREDICTION_RESULTS_PREFIX
        ),
        segment_results_bucket=os.getenv("SEGMENT_RESULTS_BUCKET", "ml-artifacts"),
        segment_results_prefix=os.getenv(
            "SEGMENT_RESULTS_PREFIX", DEFAULT_SEGMENT_RESULTS_PREFIX
        ),
    )


def get_inference_config() -> InferenceConfig:
    return InferenceConfig(
        host=os.getenv("INFERENCE_HOST", "0.0.0.0"),
        port=_int_env("INFERENCE_PORT", 8001),
        model_uri=os.getenv("MODEL_URI", DEFAULT_MODEL_URI),
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        s3_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        s3_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        segment_results_bucket=os.getenv("SEGMENT_RESULTS_BUCKET", "ml-artifacts"),
        segment_results_prefix=os.getenv(
            "SEGMENT_RESULTS_PREFIX", DEFAULT_SEGMENT_RESULTS_PREFIX
        ),
    )
