from __future__ import annotations

import io
import json

import joblib
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


class FakeArtifactStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def load_joblib(self, uri: str):
        return joblib.load(io.BytesIO(self.objects[uri]))

    def load_json(self, uri: str) -> dict:
        if uri not in self.objects:
            raise FileNotFoundError(uri)
        return json.loads(self.objects[uri].decode("utf-8"))

    def save_joblib(self, uri: str, artifact) -> None:
        buffer = io.BytesIO()
        joblib.dump(artifact, buffer)
        self.objects[uri] = buffer.getvalue()

    def copy_uri(self, source_uri: str, destination_uri: str) -> None:
        self.objects[destination_uri] = self.objects[source_uri]

    def save_json(self, uri: str, payload: dict) -> None:
        self.objects[uri] = json.dumps(payload, indent=2).encode("utf-8")


def _artifact_uris() -> dict[str, str]:
    return {
        "production_model_uri": "s3://ml-artifacts/models/production/model.joblib",
        "baseline_model_uri": "s3://ml-artifacts/models/production/baseline.joblib",
        "report_uri": "s3://ml-artifacts/models/production/training_metrics.json",
        "feature_schema_uri": "s3://ml-artifacts/models/production/feature_schema.json",
        "feature_stats_uri": "s3://ml-artifacts/models/production/feature_stats.json",
        "model_info_uri": "s3://ml-artifacts/models/production/model_info.json",
        "model_registry_uri": "s3://ml-artifacts/models/registry/model_registry.json",
    }


def test_train_main_creates_models_and_report(tmp_path, monkeypatch, capsys) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    artifact_uris = _artifact_uris()
    artifact_store = FakeArtifactStore()

    _write_dataset(train_csv, _sample_train_rows())
    _write_dataset(test_csv, _sample_test_rows())
    monkeypatch.setattr(train, "build_artifact_store", lambda: artifact_store)

    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--train-path",
            str(train_csv),
            "--test-path",
            str(test_csv),
            "--production-model-uri",
            artifact_uris["production_model_uri"],
            "--baseline-model-uri",
            artifact_uris["baseline_model_uri"],
            "--report-uri",
            artifact_uris["report_uri"],
            "--feature-schema-uri",
            artifact_uris["feature_schema_uri"],
            "--feature-stats-uri",
            artifact_uris["feature_stats_uri"],
            "--model-info-uri",
            artifact_uris["model_info_uri"],
            "--model-registry-uri",
            artifact_uris["model_registry_uri"],
            "--top-share",
            "0.5",
            "--promote",
        ],
    )

    train.main()

    for uri in artifact_uris.values():
        assert uri in artifact_store.objects

    report = json.loads(
        artifact_store.objects[artifact_uris["report_uri"]].decode("utf-8")
    )
    assert "baseline_logreg" in report
    assert "mvp_lightgbm" in report
    assert "roc_auc" in report["baseline_logreg"]

    feature_schema = json.loads(
        artifact_store.objects[artifact_uris["feature_schema_uri"]].decode("utf-8")
    )
    assert feature_schema["feature_count"] == 2
    assert feature_schema["features"][0]["name"] == "feature_a"

    feature_stats = json.loads(
        artifact_store.objects[artifact_uris["feature_stats_uri"]].decode("utf-8")
    )
    assert feature_stats["feature_count"] == 2
    assert feature_stats["features"][0]["name"] == "feature_a"
    assert feature_stats["features"][0]["mean"] is not None

    model_info = json.loads(
        artifact_store.objects[artifact_uris["model_info_uri"]].decode("utf-8")
    )
    version = model_info["version"]
    assert model_info["production_model"]["uri"].endswith(
        f"/models/versions/{version}/model.joblib"
    )
    assert model_info["feature_schema_uri"].endswith(
        f"/models/versions/{version}/feature_schema.json"
    )
    assert model_info["feature_stats_uri"].endswith(
        f"/models/versions/{version}/feature_stats.json"
    )
    assert model_info["model_registry_uri"] == artifact_uris["model_registry_uri"]
    assert model_info["top_share"] == 0.5
    assert model_info["version"]
    assert model_info["display_name"] == "Основная"
    assert model_info["promoted"] is True

    model_registry = json.loads(
        artifact_store.objects[artifact_uris["model_registry_uri"]].decode("utf-8")
    )
    assert model_registry["current_version"] == version
    assert model_registry["current_display_name"] == "Основная"
    assert model_registry["models"][0]["status"] == "production"
    assert model_registry["models"][0]["display_name"] == "Основная"
    assert (
        model_registry["models"][0]["artifacts"]["model_registry_uri"]
        == artifact_uris["model_registry_uri"]
    )
    assert (
        model_registry["models"][0]["artifacts"]["production_model_uri"].endswith(
            f"/models/versions/{version}/model.joblib"
        )
    )

    stdout = capsys.readouterr().out
    assert "baseline_logreg" in stdout


def test_evaluate_main_prints_metrics(tmp_path, monkeypatch, capsys) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    model_uri = "s3://ml-artifacts/models/baseline.joblib"
    artifact_store = FakeArtifactStore()

    _write_dataset(train_csv, _sample_train_rows())
    _write_dataset(test_csv, _sample_test_rows())

    train_frame = pd.read_csv(train_csv)
    _, train_features, train_target = train.split_features_target(train_frame)
    model = train.build_baseline_model()
    model.fit(train_features, train_target)
    artifact_store.save_joblib(model_uri, model)
    monkeypatch.setattr(evaluate, "build_artifact_store", lambda: artifact_store)

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--model-uri",
            model_uri,
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
