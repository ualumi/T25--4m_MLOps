"""Вспомогательные функции для офлайн-оценки модели."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd

from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore
from src.training.data import load_dataset, split_features_target
from src.training.metrics import (
    calculate_lift_at_top_share,
    calculate_precision_at_top_share,
    calculate_roc_auc,
)


def evaluate_binary_classifier(
    y_true: list[int], y_score: list[float], top_share: float = 0.2
) -> dict[str, float]:
    return {
        "roc_auc": calculate_roc_auc(y_true=y_true, y_score=y_score),
        "precision_at_top_share": calculate_precision_at_top_share(
            y_true=y_true,
            y_score=y_score,
            top_share=top_share,
        ),
        "lift_at_top_share": calculate_lift_at_top_share(
            y_true=y_true,
            y_score=y_score,
            top_share=top_share,
        ),
    }


def score_classifier(model: Any, features: pd.DataFrame) -> list[float]:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for ROC-AUC evaluation.")
    return [float(score) for score in model.predict_proba(features)[:, 1]]


def build_artifact_store() -> S3ArtifactStore:
    return S3ArtifactStore(
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def evaluate_saved_model(
    model_uri: str, dataset_path: str, top_share: float = 0.2
) -> dict[str, float]:
    model = build_artifact_store().load_joblib(model_uri)
    dataset = load_dataset(dataset_path)
    _, features, target = split_features_target(dataset)
    scores = score_classifier(model, features)
    return evaluate_binary_classifier(
        y_true=target.tolist(),
        y_score=scores,
        top_share=top_share,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate classifier on CSV dataset.")
    parser.add_argument(
        "--model-uri", required=True, help="S3 URI of the joblib model artifact."
    )
    parser.add_argument("--dataset-path", required=True, help="Path to CSV dataset.")
    parser.add_argument(
        "--top-share",
        type=float,
        default=0.2,
        help="Top share for ranking metrics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_saved_model(
        model_uri=args.model_uri,
        dataset_path=args.dataset_path,
        top_share=args.top_share,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
