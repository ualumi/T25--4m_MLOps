"""Canonical S3 layout helpers for artifacts, datasets, and results."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

DEFAULT_BUCKET = "ml-artifacts"

DEFAULT_DATASET_UPLOAD_PREFIX = "inference/uploads"
DEFAULT_RETRAIN_SOURCE_PREFIX = "training/uploads"
DEFAULT_RETRAIN_NON_PROMOTED_PREFIX = "training/non-promoted"
DEFAULT_PREDICTION_RESULTS_PREFIX = "inference/results"
DEFAULT_SEGMENT_RESULTS_PREFIX = "inference/segments"

DEFAULT_MODEL_URI = f"s3://{DEFAULT_BUCKET}/models/production/model.joblib"
DEFAULT_BASELINE_MODEL_URI = f"s3://{DEFAULT_BUCKET}/models/production/baseline.joblib"
DEFAULT_REPORT_URI = (
    f"s3://{DEFAULT_BUCKET}/models/production/training_metrics.json"
)
DEFAULT_FEATURE_SCHEMA_URI = (
    f"s3://{DEFAULT_BUCKET}/models/production/feature_schema.json"
)
DEFAULT_FEATURE_STATS_URI = (
    f"s3://{DEFAULT_BUCKET}/models/production/feature_stats.json"
)
DEFAULT_MODEL_INFO_URI = f"s3://{DEFAULT_BUCKET}/models/production/model_info.json"
DEFAULT_MODEL_REGISTRY_URI = (
    f"s3://{DEFAULT_BUCKET}/models/registry/model_registry.json"
)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def build_versioned_artifact_uris(
    model_registry_uri: str,
    version: str,
) -> dict[str, str]:
    bucket, registry_key = parse_s3_uri(model_registry_uri)
    models_root = _models_root_from_registry_key(registry_key)
    version_prefix = (models_root / "versions" / version).as_posix()
    return {
        "production_model_uri": f"s3://{bucket}/{version_prefix}/model.joblib",
        "baseline_model_uri": f"s3://{bucket}/{version_prefix}/baseline.joblib",
        "report_uri": f"s3://{bucket}/{version_prefix}/training_metrics.json",
        "feature_schema_uri": f"s3://{bucket}/{version_prefix}/feature_schema.json",
        "feature_stats_uri": f"s3://{bucket}/{version_prefix}/feature_stats.json",
        "model_info_uri": f"s3://{bucket}/{version_prefix}/model_info.json",
        "model_registry_uri": model_registry_uri,
    }


def rewrite_legacy_s3_uri(uri: str) -> str:
    if not uri.startswith("s3://"):
        return uri

    bucket, key = parse_s3_uri(uri)
    path = PurePosixPath(key)
    parts = path.parts

    if not parts:
        return uri

    if parts[0] == "uploads":
        return f"s3://{bucket}/{(PurePosixPath(DEFAULT_DATASET_UPLOAD_PREFIX) / PurePosixPath(*parts[1:])).as_posix()}"

    if parts[0] == "prediction-results":
        return f"s3://{bucket}/{(PurePosixPath(DEFAULT_PREDICTION_RESULTS_PREFIX) / PurePosixPath(*parts[1:])).as_posix()}"

    if parts[0] == "segments":
        return f"s3://{bucket}/{(PurePosixPath(DEFAULT_SEGMENT_RESULTS_PREFIX) / PurePosixPath(*parts[1:])).as_posix()}"

    if parts[0] == "retraining-uploads":
        return f"s3://{bucket}/{(PurePosixPath(DEFAULT_RETRAIN_SOURCE_PREFIX) / PurePosixPath(*parts[1:])).as_posix()}"

    if parts[:2] == ("datasets", "non-promoted"):
        suffix = PurePosixPath(*parts[2:]) if len(parts) > 2 else PurePosixPath()
        return f"s3://{bucket}/{(PurePosixPath(DEFAULT_RETRAIN_NON_PROMOTED_PREFIX) / suffix).as_posix()}"

    if key == "reports/training_metrics.json":
        return DEFAULT_REPORT_URI.replace(DEFAULT_BUCKET, bucket, 1)

    if key == "models/lgb_model.joblib":
        return DEFAULT_MODEL_URI.replace(DEFAULT_BUCKET, bucket, 1)

    if key == "models/baseline_logreg.joblib":
        return DEFAULT_BASELINE_MODEL_URI.replace(DEFAULT_BUCKET, bucket, 1)

    if key == "models/feature_schema.json":
        return DEFAULT_FEATURE_SCHEMA_URI.replace(DEFAULT_BUCKET, bucket, 1)

    if key == "models/feature_stats.json":
        return DEFAULT_FEATURE_STATS_URI.replace(DEFAULT_BUCKET, bucket, 1)

    if key == "models/model_info.json":
        return DEFAULT_MODEL_INFO_URI.replace(DEFAULT_BUCKET, bucket, 1)

    if key == "models/model_registry.json":
        return DEFAULT_MODEL_REGISTRY_URI.replace(DEFAULT_BUCKET, bucket, 1)

    if (
        len(parts) == 3
        and parts[0] == "models"
        and parts[1] not in {"production", "registry", "versions"}
    ):
        filename_map = {
            "lgb_model.joblib": "model.joblib",
            "baseline_logreg.joblib": "baseline.joblib",
            "training_metrics.json": "training_metrics.json",
            "feature_schema.json": "feature_schema.json",
            "feature_stats.json": "feature_stats.json",
            "model_info.json": "model_info.json",
        }
        mapped_name = filename_map.get(parts[2])
        if mapped_name is not None:
            versioned_key = PurePosixPath("models") / "versions" / parts[1] / mapped_name
            return f"s3://{bucket}/{versioned_key.as_posix()}"

    return uri


def rewrite_payload_uris(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: rewrite_payload_uris(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [rewrite_payload_uris(item) for item in payload]
    if isinstance(payload, str):
        return rewrite_legacy_s3_uri(payload)
    return payload


def _models_root_from_registry_key(registry_key: str) -> PurePosixPath:
    path = PurePosixPath(registry_key)
    if len(path.parts) >= 2 and path.parts[-2] == "registry":
        return PurePosixPath(*path.parts[:-2])
    return path.parent
