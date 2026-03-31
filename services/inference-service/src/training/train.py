"""Скрипт для офлайн-обучения и оценки моделей."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore
from src.training.data import get_feature_columns, load_dataset, split_features_target
from src.training.evaluate import evaluate_binary_classifier, score_classifier

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in CURRENT_FILE.parents if (parent / "data").exists()),
    CURRENT_FILE.parents[2],
)
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed.csv"
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "test_data.csv"
DEFAULT_PRODUCTION_MODEL_URI = "s3://ml-artifacts/models/lgb_model.joblib"
DEFAULT_BASELINE_MODEL_URI = "s3://ml-artifacts/models/baseline_logreg.joblib"
DEFAULT_REPORT_URI = "s3://ml-artifacts/reports/training_metrics.json"
DEFAULT_FEATURE_SCHEMA_URI = "s3://ml-artifacts/models/feature_schema.json"
DEFAULT_MODEL_INFO_URI = "s3://ml-artifacts/models/model_info.json"


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


def build_model_info(
    production_model: Any,
    baseline_model: Any,
    metrics: dict[str, dict[str, float]],
    train_path: str | Path,
    test_path: str | Path,
    top_share: float,
    production_model_uri: str,
    baseline_model_uri: str,
    report_uri: str,
    feature_schema_uri: str,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "top_share": top_share,
        "production_model": {
            "type": type(production_model).__name__,
            "uri": production_model_uri,
        },
        "baseline_model": {
            "type": type(baseline_model).__name__,
            "uri": baseline_model_uri,
        },
        "report_uri": report_uri,
        "feature_schema_uri": feature_schema_uri,
        "metrics_summary": metrics,
    }


def save_training_outputs(
    production_model: Any,
    baseline_model: Any,
    metrics: dict[str, dict[str, float]],
    production_model_uri: str,
    baseline_model_uri: str,
    report_uri: str,
    feature_schema: dict[str, Any],
    feature_schema_uri: str,
    model_info: dict[str, Any],
    model_info_uri: str,
) -> None:
    artifact_store = build_artifact_store()
    artifact_store.save_joblib(production_model_uri, production_model)
    artifact_store.save_joblib(baseline_model_uri, baseline_model)
    artifact_store.save_json(report_uri, metrics)
    artifact_store.save_json(feature_schema_uri, feature_schema)
    artifact_store.save_json(model_info_uri, model_info)


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
        "--model-info-uri",
        default=DEFAULT_MODEL_INFO_URI,
        help="S3 URI for the model metadata JSON.",
    )
    parser.add_argument(
        "--top-share",
        type=float,
        default=0.2,
        help="Top share used for ranking metrics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
    model_info = build_model_info(
        production_model=mvp_model,
        baseline_model=baseline_model,
        metrics=metrics,
        train_path=args.train_path,
        test_path=args.test_path,
        top_share=args.top_share,
        production_model_uri=args.production_model_uri,
        baseline_model_uri=args.baseline_model_uri,
        report_uri=args.report_uri,
        feature_schema_uri=args.feature_schema_uri,
    )
    save_training_outputs(
        production_model=mvp_model,
        baseline_model=baseline_model,
        metrics=metrics,
        production_model_uri=args.production_model_uri,
        baseline_model_uri=args.baseline_model_uri,
        report_uri=args.report_uri,
        feature_schema=feature_schema,
        feature_schema_uri=args.feature_schema_uri,
        model_info=model_info,
        model_info_uri=args.model_info_uri,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
