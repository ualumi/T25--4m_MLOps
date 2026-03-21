from __future__ import annotations

import json

import pandas as pd

from src.training import evaluate, train


def _write_dataset(path, rows: list[dict[str, float | int]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _sample_train_rows() -> list[dict[str, float | int]]:
    return [
        {"user_id": 1, "feature_a": 0.05, "feature_b": 0.10, "churn": 0},
        {"user_id": 2, "feature_a": 0.10, "feature_b": 0.15, "churn": 0},
        {"user_id": 3, "feature_a": 0.15, "feature_b": 0.20, "churn": 0},
        {"user_id": 4, "feature_a": 0.20, "feature_b": 0.25, "churn": 0},
        {"user_id": 5, "feature_a": 0.80, "feature_b": 0.75, "churn": 1},
        {"user_id": 6, "feature_a": 0.85, "feature_b": 0.80, "churn": 1},
        {"user_id": 7, "feature_a": 0.90, "feature_b": 0.85, "churn": 1},
        {"user_id": 8, "feature_a": 0.95, "feature_b": 0.90, "churn": 1},
    ]


def _sample_test_rows() -> list[dict[str, float | int]]:
    return [
        {"user_id": 101, "feature_a": 0.12, "feature_b": 0.12, "churn": 0},
        {"user_id": 102, "feature_a": 0.18, "feature_b": 0.20, "churn": 0},
        {"user_id": 103, "feature_a": 0.82, "feature_b": 0.78, "churn": 1},
        {"user_id": 104, "feature_a": 0.91, "feature_b": 0.88, "churn": 1},
    ]


def test_train_main_creates_models_and_report(tmp_path, monkeypatch, capsys) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    model_dir = tmp_path / "models"
    report_path = tmp_path / "reports" / "metrics.json"

    _write_dataset(train_csv, _sample_train_rows())
    _write_dataset(test_csv, _sample_test_rows())

    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--train-path",
            str(train_csv),
            "--test-path",
            str(test_csv),
            "--model-dir",
            str(model_dir),
            "--report-path",
            str(report_path),
            "--top-share",
            "0.5",
        ],
    )

    train.main()

    assert (model_dir / "baseline_logreg.joblib").exists()
    assert not (model_dir / "lgb_model_candidate.joblib").exists()
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "baseline_logreg" in report
    assert "mvp_lightgbm" in report
    assert "roc_auc" in report["baseline_logreg"]

    stdout = capsys.readouterr().out
    assert "baseline_logreg" in stdout


def test_evaluate_main_prints_metrics(tmp_path, monkeypatch, capsys) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    model_path = tmp_path / "baseline.joblib"

    _write_dataset(train_csv, _sample_train_rows())
    _write_dataset(test_csv, _sample_test_rows())

    train_frame = pd.read_csv(train_csv)
    _, train_features, train_target = train.split_features_target(train_frame)
    model = train.build_baseline_model()
    model.fit(train_features, train_target)
    train.joblib.dump(model, model_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--model-path",
            str(model_path),
            "--dataset-path",
            str(test_csv),
            "--top-share",
            "0.5",
        ],
    )

    evaluate.main()

    stdout = capsys.readouterr().out
    metrics = json.loads(stdout)
    assert "roc_auc" in metrics
    assert "precision_at_top_share" in metrics
    assert "lift_at_top_share" in metrics
