"""Скрипт для офлайн-обучения и оценки моделей."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.s3_layout import (
    DEFAULT_BASELINE_MODEL_URI,
    DEFAULT_FEATURE_SCHEMA_URI,
    DEFAULT_FEATURE_STATS_URI,
    DEFAULT_MODEL_INFO_URI,
    DEFAULT_MODEL_REGISTRY_URI,
    DEFAULT_MODEL_URI,
    DEFAULT_REPORT_URI,
    build_versioned_artifact_uris,
)
from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore
from src.training.data import get_feature_columns, load_dataset, split_features_target
from src.training.evaluate import evaluate_binary_classifier, score_classifier
from src.training.model_registry import (
    build_model_registry_entry,
    load_model_registry,
    upsert_model_entry,
)

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in CURRENT_FILE.parents if (parent / "data").exists()),
    CURRENT_FILE.parents[2],
)
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed.csv"
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "test_data.csv"
DEFAULT_PRODUCTION_MODEL_URI = DEFAULT_MODEL_URI


@dataclass(frozen=True)
class TrainingArtifactUris:
    production_model_uri: str
    baseline_model_uri: str
    report_uri: str
    feature_schema_uri: str
    feature_stats_uri: str
    model_info_uri: str
    model_registry_uri: str


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class TrainingRunContext:
    train_path: str | Path
    test_path: str | Path
    top_share: float
    artifact_uris: TrainingArtifactUris
    stable_production_model_uri: str
    stable_baseline_model_uri: str
    stable_report_uri: str
    stable_feature_schema_uri: str
    stable_feature_stats_uri: str
    stable_model_info_uri: str
    generated_at: str
    version: str
    promote: bool
    source_dataset_uris: list[str]


def build_baseline_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_mvp_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )


def train_and_evaluate_model(
    model: Any,
    train_path: str | Path,
    test_path: str | Path,
    top_share: float = 0.2,
) -> tuple[Any, dict[str, float]]:
    train_frame = load_dataset(train_path)
    test_frame = load_dataset(test_path)

    _, train_features, train_target = split_features_target(train_frame)
    _, test_features, test_target = split_features_target(test_frame)

    model.fit(train_features, train_target)
    scores = score_classifier(model, test_features)
    metrics = evaluate_binary_classifier(
        y_true=test_target.tolist(),
        y_score=scores,
        top_share=top_share,
    )
    return model, metrics


def build_artifact_store() -> S3ArtifactStore:
    return S3ArtifactStore(
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        sse_mode=os.getenv("S3_SSE_MODE"),
        sse_kms_key_id=os.getenv("S3_SSE_KMS_KEY_ID"),
    )


def build_feature_schema(dataset_path: str | Path) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    feature_columns = get_feature_columns(dataset)
    return {
        "feature_count": len(feature_columns),
        "id_column": "user_id",
        "target_column": "churn",
        "features": [
            {
                "name": column,
                "dtype": str(dataset[column].dtype),
            }
            for column in feature_columns
        ],
    }


def _safe_numeric_stat(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def build_feature_stats(dataset_path: str | Path) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    feature_columns = get_feature_columns(dataset)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "feature_count": len(feature_columns),
        "features": [
            {
                "name": column,
                "dtype": str(dataset[column].dtype),
                "null_count": int(dataset[column].isna().sum()),
                "mean": _safe_numeric_stat(dataset[column].mean()),
                "std": _safe_numeric_stat(dataset[column].std()),
                "min": _safe_numeric_stat(dataset[column].min()),
                "max": _safe_numeric_stat(dataset[column].max()),
            }
            for column in feature_columns
        ],
    }


def _build_display_name(generated_at: str, *, promoted: bool) -> str:
    if promoted:
        return "Основная"
    date_part = generated_at.split("T", maxsplit=1)[0]
    try:
        parsed = datetime.fromisoformat(date_part)
    except ValueError:
        return date_part
    return parsed.strftime("%d.%m.%Y")


def build_model_info(
    production_model: Any,
    baseline_model: Any,
    metrics: dict[str, dict[str, float]],
    run_context: TrainingRunContext,
) -> dict[str, Any]:
    return {
        "generated_at": run_context.generated_at,
        "version": run_context.version,
        "display_name": _build_display_name(
            run_context.generated_at,
            promoted=run_context.promote,
        ),
        "train_path": str(run_context.train_path),
        "test_path": str(run_context.test_path),
        "top_share": run_context.top_share,
        "promoted": run_context.promote,
        "source_dataset_uris": run_context.source_dataset_uris,
        "production_model": {
            "type": type(production_model).__name__,
            "uri": run_context.artifact_uris.production_model_uri,
        },
        "baseline_model": {
            "type": type(baseline_model).__name__,
            "uri": run_context.artifact_uris.baseline_model_uri,
        },
        "report_uri": run_context.artifact_uris.report_uri,
        "feature_schema_uri": run_context.artifact_uris.feature_schema_uri,
        "feature_stats_uri": run_context.artifact_uris.feature_stats_uri,
        "model_registry_uri": run_context.artifact_uris.model_registry_uri,
        "metrics_summary": metrics,
    }


def build_model_registry(
    production_model: Any,
    baseline_model: Any,
    metrics: dict[str, dict[str, float]],
    feature_schema: dict[str, Any],
    run_context: TrainingRunContext,
) -> dict[str, Any]:
    artifact_store = build_artifact_store()
    existing_registry = load_model_registry(
        artifact_store,
        run_context.artifact_uris.model_registry_uri,
    )
    new_entry = build_model_registry_entry(
        version=run_context.version,
        status="production" if run_context.promote else "candidate",
        generated_at=run_context.generated_at,
        top_share=run_context.top_share,
        feature_count=feature_schema["feature_count"],
        artifacts={
            "production_model_uri": run_context.artifact_uris.production_model_uri,
            "baseline_model_uri": run_context.artifact_uris.baseline_model_uri,
            "report_uri": run_context.artifact_uris.report_uri,
            "feature_schema_uri": run_context.artifact_uris.feature_schema_uri,
            "feature_stats_uri": run_context.artifact_uris.feature_stats_uri,
            "model_info_uri": run_context.artifact_uris.model_info_uri,
            "model_registry_uri": run_context.artifact_uris.model_registry_uri,
        },
        metrics_summary=metrics,
        production_model_type=type(production_model).__name__,
        baseline_model_type=type(baseline_model).__name__,
        source_dataset_uris=run_context.source_dataset_uris,
    )
    return upsert_model_entry(
        existing_registry,
        new_entry,
        promote=run_context.promote,
    )


# pylint: disable=too-many-arguments
def save_training_outputs(
    production_model: Any,
    baseline_model: Any,
    metrics: dict[str, dict[str, float]],
    feature_schema: dict[str, Any],
    feature_stats: dict[str, Any],
    model_info: dict[str, Any],
    model_registry: dict[str, Any],
    artifact_uris: TrainingArtifactUris,
    run_context: TrainingRunContext,
) -> None:
    artifact_store = build_artifact_store()
    artifact_store.save_joblib(artifact_uris.production_model_uri, production_model)
    artifact_store.save_joblib(artifact_uris.baseline_model_uri, baseline_model)
    artifact_store.save_json(artifact_uris.report_uri, metrics)
    artifact_store.save_json(artifact_uris.feature_schema_uri, feature_schema)
    artifact_store.save_json(artifact_uris.feature_stats_uri, feature_stats)
    artifact_store.save_json(artifact_uris.model_info_uri, model_info)
    artifact_store.save_json(artifact_uris.model_registry_uri, model_registry)
    if run_context.promote:
        artifact_store.copy_uri(
            artifact_uris.production_model_uri,
            run_context.stable_production_model_uri,
        )
        artifact_store.copy_uri(
            artifact_uris.baseline_model_uri,
            run_context.stable_baseline_model_uri,
        )
        artifact_store.copy_uri(
            artifact_uris.report_uri,
            run_context.stable_report_uri,
        )
        artifact_store.copy_uri(
            artifact_uris.feature_schema_uri,
            run_context.stable_feature_schema_uri,
        )
        artifact_store.copy_uri(
            artifact_uris.feature_stats_uri,
            run_context.stable_feature_stats_uri,
        )
        artifact_store.copy_uri(
            artifact_uris.model_info_uri,
            run_context.stable_model_info_uri,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train churn models on processed data."
    )
    parser.add_argument(
        "--train-path",
        default=str(DEFAULT_TRAIN_PATH),
        help="Path to training CSV dataset.",
    )
    parser.add_argument(
        "--test-path",
        default=str(DEFAULT_TEST_PATH),
        help="Path to evaluation CSV dataset.",
    )
    parser.add_argument(
        "--production-model-uri",
        default=DEFAULT_PRODUCTION_MODEL_URI,
        help="S3 URI for the production LightGBM model.",
    )
    parser.add_argument(
        "--baseline-model-uri",
        default=DEFAULT_BASELINE_MODEL_URI,
        help="S3 URI for the baseline model artifact.",
    )
    parser.add_argument(
        "--report-uri",
        default=DEFAULT_REPORT_URI,
        help="S3 URI for the evaluation metrics JSON.",
    )
    parser.add_argument(
        "--feature-schema-uri",
        default=DEFAULT_FEATURE_SCHEMA_URI,
        help="S3 URI for the feature schema JSON.",
    )
    parser.add_argument(
        "--feature-stats-uri",
        default=DEFAULT_FEATURE_STATS_URI,
        help="S3 URI for the feature statistics JSON.",
    )
    parser.add_argument(
        "--model-info-uri",
        default=DEFAULT_MODEL_INFO_URI,
        help="S3 URI for the model metadata JSON.",
    )
    parser.add_argument(
        "--model-registry-uri",
        default=DEFAULT_MODEL_REGISTRY_URI,
        help="S3 URI for the model registry JSON.",
    )
    parser.add_argument(
        "--top-share",
        type=float,
        default=0.2,
        help="Top share used for ranking metrics.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Explicit model version. Defaults to current UTC timestamp.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Mark current version as production in model registry.",
    )
    parser.add_argument(
        "--source-dataset-uri",
        action="append",
        default=[],
        help="S3 URI of source dataset used for this training run. Can be repeated.",
    )
    return parser


def _build_versioned_artifact_uris(
    model_registry_uri: str,
    version: str,
) -> TrainingArtifactUris:
    artifact_uris = build_versioned_artifact_uris(model_registry_uri, version)
    return TrainingArtifactUris(
        production_model_uri=artifact_uris["production_model_uri"],
        baseline_model_uri=artifact_uris["baseline_model_uri"],
        report_uri=artifact_uris["report_uri"],
        feature_schema_uri=artifact_uris["feature_schema_uri"],
        feature_stats_uri=artifact_uris["feature_stats_uri"],
        model_info_uri=artifact_uris["model_info_uri"],
        model_registry_uri=artifact_uris["model_registry_uri"],
    )


def main() -> None:
    args = build_parser().parse_args()
    generated_at = datetime.now(timezone.utc).isoformat()
    version = args.version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_uris = _build_versioned_artifact_uris(
        model_registry_uri=args.model_registry_uri,
        version=version,
    )
    run_context = TrainingRunContext(
        train_path=args.train_path,
        test_path=args.test_path,
        top_share=args.top_share,
        artifact_uris=artifact_uris,
        stable_production_model_uri=args.production_model_uri,
        stable_baseline_model_uri=args.baseline_model_uri,
        stable_report_uri=args.report_uri,
        stable_feature_schema_uri=args.feature_schema_uri,
        stable_feature_stats_uri=args.feature_stats_uri,
        stable_model_info_uri=args.model_info_uri,
        generated_at=generated_at,
        version=version,
        promote=args.promote,
        source_dataset_uris=list(args.source_dataset_uri),
    )

    baseline_model, baseline_metrics = train_and_evaluate_model(
        model=build_baseline_model(),
        train_path=args.train_path,
        test_path=args.test_path,
        top_share=args.top_share,
    )
    mvp_model, mvp_metrics = train_and_evaluate_model(
        model=build_mvp_model(),
        train_path=args.train_path,
        test_path=args.test_path,
        top_share=args.top_share,
    )

    metrics = {
        "baseline_logreg": baseline_metrics,
        "mvp_lightgbm": mvp_metrics,
    }
    feature_schema = build_feature_schema(args.train_path)
    feature_stats = build_feature_stats(args.train_path)
    model_info = build_model_info(
        production_model=mvp_model,
        baseline_model=baseline_model,
        metrics=metrics,
        run_context=run_context,
    )
    model_registry = build_model_registry(
        production_model=mvp_model,
        baseline_model=baseline_model,
        metrics=metrics,
        feature_schema=feature_schema,
        run_context=run_context,
    )
    save_training_outputs(
        production_model=mvp_model,
        baseline_model=baseline_model,
        metrics=metrics,
        feature_schema=feature_schema,
        feature_stats=feature_stats,
        model_info=model_info,
        model_registry=model_registry,
        artifact_uris=artifact_uris,
        run_context=run_context,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
