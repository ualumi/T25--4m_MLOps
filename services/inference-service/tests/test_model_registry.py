from src.training.model_registry import (
    build_empty_registry,
    build_model_registry_entry,
    upsert_model_entry,
)


def _entry(version: str, status: str = "candidate", roc_auc: float = 0.7) -> dict:
    return build_model_registry_entry(
        version=version,
        status=status,
        generated_at=f"2026-04-04T00:00:0{version[-1]}Z",
        top_share=0.2,
        feature_count=28,
        artifacts={
            "production_model_uri": (
                f"s3://ml-artifacts/models/versions/{version}/model.joblib"
            ),
            "baseline_model_uri": (
                f"s3://ml-artifacts/models/versions/{version}/baseline.joblib"
            ),
            "report_uri": (
                f"s3://ml-artifacts/models/versions/{version}/training_metrics.json"
            ),
            "feature_schema_uri": (
                f"s3://ml-artifacts/models/versions/{version}/feature_schema.json"
            ),
            "feature_stats_uri": (
                f"s3://ml-artifacts/models/versions/{version}/feature_stats.json"
            ),
            "model_info_uri": (
                f"s3://ml-artifacts/models/versions/{version}/model_info.json"
            ),
            "model_registry_uri": (
                "s3://ml-artifacts/models/registry/model_registry.json"
            ),
        },
        metrics_summary={
            "baseline_logreg": {"roc_auc": 0.6},
            "mvp_lightgbm": {"roc_auc": roc_auc},
        },
        production_model_type="LGBMClassifier",
        baseline_model_type="Pipeline",
        source_dataset_uris=[f"s3://ml-artifacts/training/uploads/{version}.csv"],
    )


def test_upsert_model_entry_keeps_candidate_without_promotion() -> None:
    registry = build_empty_registry()

    updated = upsert_model_entry(registry, _entry("v1"), promote=False)

    assert updated["current_version"] is None
    assert updated["current_display_name"] is None
    assert updated["models"][0]["status"] == "candidate"
    assert updated["models"][0]["display_name"] == "04.04.2026"


def test_upsert_model_entry_promotes_new_model_and_archives_previous() -> None:
    registry = upsert_model_entry(build_empty_registry(), _entry("v1"), promote=True)

    updated = upsert_model_entry(registry, _entry("v2"), promote=True)

    assert updated["current_version"] == "v2"
    assert updated["current_display_name"] == "Основная"
    statuses = {entry["version"]: entry["status"] for entry in updated["models"]}
    display_names = {
        entry["version"]: entry["display_name"] for entry in updated["models"]
    }
    assert statuses["v2"] == "production"
    assert statuses["v1"] == "archived"
    assert display_names["v2"] == "Основная"
    assert display_names["v1"] == "04.04.2026"
