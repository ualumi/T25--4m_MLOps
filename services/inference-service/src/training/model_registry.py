"""Helpers for persisting and updating model registry state."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore

PROMOTION_METRIC = "roc_auc"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_name_from_generated_at(generated_at: str) -> str:
    try:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return generated_at
    return parsed.strftime("%d.%m.%Y")


def build_empty_registry() -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "promotion_metric": PROMOTION_METRIC,
        "current_version": None,
        "current_display_name": None,
        "models": [],
    }


def _normalize_model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    metrics_summary = entry.get("metrics_summary", {})
    mvp_metrics = metrics_summary.get("mvp_lightgbm", {})
    source_dataset_uris = entry.get("source_dataset_uris", [])
    return {
        "version": entry.get("version"),
        "status": entry.get("status", "archived"),
        "generated_at": entry.get("generated_at", _now_iso()),
        "display_name": entry.get(
            "display_name",
            "Основная"
            if entry.get("status") == "production"
            else _display_name_from_generated_at(entry.get("generated_at", _now_iso())),
        ),
        "top_share": entry.get("top_share"),
        "feature_count": entry.get("feature_count"),
        "source_dataset_uris": list(source_dataset_uris),
        "promotion_metric": entry.get("promotion_metric", PROMOTION_METRIC),
        "promotion_metric_value": entry.get(
            "promotion_metric_value",
            mvp_metrics.get(PROMOTION_METRIC),
        ),
        "artifacts": deepcopy(entry.get("artifacts", {})),
        "models": deepcopy(entry.get("models", {})),
        "metrics_summary": deepcopy(metrics_summary),
    }


def normalize_registry(payload: dict[str, Any]) -> dict[str, Any]:
    registry = build_empty_registry()
    registry["generated_at"] = payload.get("generated_at", registry["generated_at"])
    registry["promotion_metric"] = payload.get(
        "promotion_metric", registry["promotion_metric"]
    )
    registry["models"] = [
        _normalize_model_entry(entry)
        for entry in payload.get("models", [])
        if isinstance(entry, dict)
    ]
    registry["current_version"] = payload.get("current_version")

    if registry["current_version"] is None:
        for entry in registry["models"]:
            if entry["status"] == "production":
                registry["current_version"] = entry["version"]
                break
    current_entry = next(
        (
            entry
            for entry in registry["models"]
            if entry["version"] == registry["current_version"]
        ),
        None,
    )
    registry["current_display_name"] = (
        current_entry["display_name"] if current_entry is not None else None
    )
    return registry


def load_model_registry(artifact_store: S3ArtifactStore, uri: str) -> dict[str, Any]:
    try:
        payload = artifact_store.load_json(uri)
    except FileNotFoundError:
        return build_empty_registry()
    return normalize_registry(payload)


def build_model_registry_entry(
    *,
    version: str,
    status: str,
    generated_at: str,
    top_share: float,
    feature_count: int,
    artifacts: dict[str, str],
    metrics_summary: dict[str, dict[str, float]],
    production_model_type: str,
    baseline_model_type: str,
    source_dataset_uris: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": version,
        "status": status,
        "generated_at": generated_at,
        "display_name": "Основная"
        if status == "production"
        else _display_name_from_generated_at(generated_at),
        "top_share": top_share,
        "feature_count": feature_count,
        "source_dataset_uris": list(source_dataset_uris or []),
        "promotion_metric": PROMOTION_METRIC,
        "promotion_metric_value": metrics_summary["mvp_lightgbm"][PROMOTION_METRIC],
        "artifacts": deepcopy(artifacts),
        "models": {
            "production_type": production_model_type,
            "baseline_type": baseline_model_type,
        },
        "metrics_summary": deepcopy(metrics_summary),
    }


def upsert_model_entry(
    registry: dict[str, Any],
    new_entry: dict[str, Any],
    *,
    promote: bool,
) -> dict[str, Any]:
    normalized = normalize_registry(registry)
    models = [
        entry for entry in normalized["models"] if entry["version"] != new_entry["version"]
    ]

    if promote:
        for entry in models:
            if entry["status"] == "production":
                entry["status"] = "archived"
                entry["display_name"] = _display_name_from_generated_at(
                    entry["generated_at"]
                )
        new_entry["status"] = "production"
        new_entry["display_name"] = "Основная"
        normalized["current_version"] = new_entry["version"]
    else:
        new_entry["status"] = new_entry.get("status", "candidate")
        if new_entry["status"] != "production":
            new_entry["display_name"] = _display_name_from_generated_at(
                new_entry["generated_at"]
            )
        if normalized["current_version"] is None and new_entry["status"] == "production":
            normalized["current_version"] = new_entry["version"]

    models.append(_normalize_model_entry(new_entry))
    normalized["models"] = sorted(
        models,
        key=lambda entry: str(entry["generated_at"]),
        reverse=True,
    )
    normalized["generated_at"] = _now_iso()
    current_entry = next(
        (
            entry
            for entry in normalized["models"]
            if entry["version"] == normalized["current_version"]
        ),
        None,
    )
    normalized["current_display_name"] = (
        current_entry["display_name"] if current_entry is not None else None
    )
    return normalized
