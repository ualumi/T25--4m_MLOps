"""Скрипт для офлайн-обучения baseline- и MVP-моделей."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.data import load_dataset, split_features_target
from src.training.evaluate import evaluate_binary_classifier, score_classifier

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed.csv"
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "test_data.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "training_metrics.json"


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


def save_training_outputs(
    baseline_model: Any,
    metrics: dict[str, dict[str, float]],
    model_dir: str | Path,
    report_path: str | Path,
) -> None:
    model_dir = Path(model_dir)
    report_path = Path(report_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(baseline_model, model_dir / "baseline_logreg.joblib")
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train churn models on processed data.")
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
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to store trained models.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="JSON file for evaluation metrics.",
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
    save_training_outputs(
        baseline_model=baseline_model,
        metrics=metrics,
        model_dir=args.model_dir,
        report_path=args.report_path,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
