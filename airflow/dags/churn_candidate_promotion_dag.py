from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

PROJECT_ROOT = Path("/opt/project")
INFERENCE_SERVICE_ROOT = PROJECT_ROOT / "services" / "inference-service"

for candidate_path in (PROJECT_ROOT, INFERENCE_SERVICE_ROOT):
    if str(candidate_path) not in sys.path:
        sys.path.append(str(candidate_path))

from config.s3_layout import (  # noqa: E402
    DEFAULT_BASELINE_MODEL_URI,
    DEFAULT_FEATURE_SCHEMA_URI,
    DEFAULT_FEATURE_STATS_URI,
    DEFAULT_MODEL_INFO_URI,
    DEFAULT_MODEL_REGISTRY_URI,
    DEFAULT_MODEL_URI,
    DEFAULT_REPORT_URI,
    DEFAULT_RETRAIN_NON_PROMOTED_PREFIX,
    DEFAULT_RETRAIN_SOURCE_PREFIX,
    build_versioned_artifact_uris,
)
from src.infrastructure.storage.s3_artifact_store import (  # noqa: E402
    S3ArtifactStore,
    parse_s3_uri,
)
from src.training.data import TARGET_COLUMN, get_feature_columns, load_dataset  # noqa: E402
from src.training.evaluate import evaluate_saved_model  # noqa: E402
from src.training.model_registry import load_model_registry, upsert_model_entry  # noqa: E402

INFERENCE_BASE_URL = os.getenv("AIRFLOW_INFERENCE_BASE_URL", "http://inference-service:8001")
SOURCE_DATASETS_BUCKET = os.getenv("RETRAIN_SOURCE_DATASETS_BUCKET") or os.getenv(
    "RETRAIN_SOURCE_DATASETS_BUCKET",
    "ml-artifacts",
)
SOURCE_DATASETS_PREFIX = os.getenv("RETRAIN_SOURCE_DATASETS_PREFIX") or os.getenv(
    "RETRAIN_SOURCE_DATASETS_PREFIX",
    DEFAULT_RETRAIN_SOURCE_PREFIX,
)
NON_PROMOTED_PREFIX = os.getenv(
    "RETRAIN_NON_PROMOTED_PREFIX",
    DEFAULT_RETRAIN_NON_PROMOTED_PREFIX,
)
REFERENCE_SCHEMA_PATH = os.getenv(
    "RETRAIN_REFERENCE_SCHEMA_PATH",
    str(PROJECT_ROOT / "data" / "processed.csv"),
)
TEST_DATASET_PATH = os.getenv(
    "RETRAIN_TEST_DATASET_PATH",
    str(PROJECT_ROOT / "data" / "test_data.csv"),
)
TOP_SHARE = float(os.getenv("RETRAIN_TOP_SHARE", "0.2"))
MODEL_URI = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
BASELINE_MODEL_URI = os.getenv(
    "BASELINE_MODEL_URI",
    DEFAULT_BASELINE_MODEL_URI,
)
REPORT_URI = os.getenv(
    "REPORT_URI",
    DEFAULT_REPORT_URI,
)
FEATURE_SCHEMA_URI = os.getenv(
    "FEATURE_SCHEMA_URI",
    DEFAULT_FEATURE_SCHEMA_URI,
)
FEATURE_STATS_URI = os.getenv(
    "FEATURE_STATS_URI",
    DEFAULT_FEATURE_STATS_URI,
)
MODEL_INFO_URI = os.getenv(
    "MODEL_INFO_URI",
    DEFAULT_MODEL_INFO_URI,
)
MODEL_REGISTRY_URI = os.getenv(
    "MODEL_REGISTRY_URI",
    DEFAULT_MODEL_REGISTRY_URI,
)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _artifact_store() -> S3ArtifactStore:
    return S3ArtifactStore(
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _request(url: str, method: str = "GET") -> int:
    request = Request(url=url, method=method)
    with urlopen(request, timeout=10) as response:
        return response.getcode()


def _wait_for_inference() -> None:
    deadline = time.time() + 180
    last_error: str | None = None

    while time.time() < deadline:
        try:
            status_code = _request(f"{INFERENCE_BASE_URL}/health")
            if status_code == 200:
                return
        except (HTTPError, URLError) as exc:
            last_error = str(exc)
        time.sleep(5)

    raise RuntimeError(
        "Inference service did not become ready before timeout."
        f" Last error: {last_error or 'unknown error'}"
    )


def _load_reference_columns() -> list[str]:
    dataset = load_dataset(REFERENCE_SCHEMA_PATH)
    return get_feature_columns(dataset)


def _is_training_dataset(uri: str) -> bool:
    return uri.lower().endswith(".csv")


def _validate_training_dataset(raw_bytes: bytes, required_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw_bytes))
    expected_columns = ["user_id", *required_columns, TARGET_COLUMN]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            "Dataset columns must match training schema exactly: "
            + ", ".join(expected_columns)
        )
    if frame.empty:
        raise ValueError("Training dataset must contain at least one row.")
    churn_values = set(frame[TARGET_COLUMN].dropna().astype(int).tolist())
    if not churn_values.issubset({0, 1}):
        raise ValueError("Target column 'churn' must contain only 0/1 values.")
    return frame


def _build_candidate_artifact_uris(version: str) -> dict[str, str]:
    return build_versioned_artifact_uris(MODEL_REGISTRY_URI, version)


def _non_promoted_destination_uri(source_uri: str, version: str) -> str:
    bucket, key = parse_s3_uri(source_uri)
    original_name = Path(key).name
    return f"s3://{bucket}/{NON_PROMOTED_PREFIX.strip('/')}/{version}/{original_name}"


@dag(
    dag_id="churn_candidate_promotion_pipeline",
    description="Train candidate model from uploaded datasets and promote only if ROC-AUC improves.",
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    schedule=os.getenv("AIRFLOW_PROMOTION_SCHEDULE", "0 0 1 * *"),
    catchup=False,
    tags=["mlops", "training", "promotion", "churn"],
)
def churn_candidate_promotion_pipeline():
    @task
    def wait_for_inference() -> None:
        _wait_for_inference()

    @task
    def prepare_candidate_dataset() -> dict[str, object]:
        version = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        artifact_store = _artifact_store()
        registry = load_model_registry(artifact_store, MODEL_REGISTRY_URI)
        processed_uris = {
            dataset_uri
            for entry in registry["models"]
            for dataset_uri in entry.get("source_dataset_uris", [])
        }
        required_columns = _load_reference_columns()
        source_uris = artifact_store.list_uris(
            SOURCE_DATASETS_BUCKET,
            SOURCE_DATASETS_PREFIX,
        )

        valid_frames: list[pd.DataFrame] = []
        valid_uris: list[str] = []
        invalid_uris: list[str] = []

        for uri in source_uris:
            if uri in processed_uris or not _is_training_dataset(uri):
                continue
            try:
                frame = _validate_training_dataset(
                    artifact_store.download_bytes(uri),
                    required_columns=required_columns,
                )
            except Exception:
                invalid_uris.append(uri)
                continue
            valid_frames.append(frame)
            valid_uris.append(uri)

        if not valid_frames:
            raise AirflowSkipException("No new valid training datasets found.")

        merged_frame = pd.concat(valid_frames, ignore_index=True)
        if len(set(merged_frame[TARGET_COLUMN].astype(int).tolist())) < 2:
            raise AirflowSkipException(
                "Merged candidate dataset does not contain both target classes."
            )

        candidate_train_path = Path(tempfile.gettempdir()) / f"candidate-train-{version}.csv"
        merged_frame.to_csv(candidate_train_path, index=False)

        return {
            "version": version,
            "candidate_train_path": str(candidate_train_path),
            "source_dataset_uris": valid_uris,
            "invalid_dataset_uris": invalid_uris,
            "artifact_uris": _build_candidate_artifact_uris(version),
        }

    @task
    def train_candidate_model(payload: dict[str, object]) -> dict[str, object]:
        artifact_uris = payload["artifact_uris"]
        command = [
            sys.executable,
            "-m",
            "src.training.train",
            "--train-path",
            str(payload["candidate_train_path"]),
            "--test-path",
            TEST_DATASET_PATH,
            "--production-model-uri",
            MODEL_URI,
            "--baseline-model-uri",
            BASELINE_MODEL_URI,
            "--report-uri",
            REPORT_URI,
            "--feature-schema-uri",
            FEATURE_SCHEMA_URI,
            "--feature-stats-uri",
            FEATURE_STATS_URI,
            "--model-info-uri",
            MODEL_INFO_URI,
            "--model-registry-uri",
            MODEL_REGISTRY_URI,
            "--top-share",
            str(TOP_SHARE),
            "--version",
            str(payload["version"]),
        ]
        for uri in payload["source_dataset_uris"]:
            command.extend(["--source-dataset-uri", str(uri)])

        subprocess.run(
            command,
            check=True,
            cwd=INFERENCE_SERVICE_ROOT,
        )
        return payload

    @task
    def compare_candidate_to_production(payload: dict[str, object]) -> dict[str, object]:
        artifact_store = _artifact_store()
        registry = load_model_registry(artifact_store, MODEL_REGISTRY_URI)
        candidate_metrics = evaluate_saved_model(
            model_uri=payload["artifact_uris"]["production_model_uri"],
            dataset_path=TEST_DATASET_PATH,
            top_share=TOP_SHARE,
        )
        current_version = registry.get("current_version")
        if current_version is None:
            production_metrics = None
            promote = True
        else:
            production_metrics = evaluate_saved_model(
                model_uri=MODEL_URI,
                dataset_path=TEST_DATASET_PATH,
                top_share=TOP_SHARE,
            )
            promote = candidate_metrics["roc_auc"] > production_metrics["roc_auc"]

        return {
            **payload,
            "promote": promote,
            "candidate_metrics": candidate_metrics,
            "production_metrics": production_metrics,
            "current_version": current_version,
        }

    @task
    def apply_promotion_decision(payload: dict[str, object]) -> dict[str, object]:
        artifact_store = _artifact_store()
        registry = load_model_registry(artifact_store, MODEL_REGISTRY_URI)
        version = str(payload["version"])
        candidate_entry = next(
            entry for entry in registry["models"] if entry["version"] == version
        )

        if payload["promote"]:
            artifact_store.copy_uri(payload["artifact_uris"]["production_model_uri"], MODEL_URI)
            artifact_store.copy_uri(
                payload["artifact_uris"]["baseline_model_uri"],
                BASELINE_MODEL_URI,
            )
            artifact_store.copy_uri(
                payload["artifact_uris"]["report_uri"],
                REPORT_URI,
            )
            artifact_store.copy_uri(
                payload["artifact_uris"]["feature_schema_uri"],
                FEATURE_SCHEMA_URI,
            )
            artifact_store.copy_uri(
                payload["artifact_uris"]["feature_stats_uri"],
                FEATURE_STATS_URI,
            )
            stable_model_info = artifact_store.load_json(
                payload["artifact_uris"]["model_info_uri"]
            )
            stable_model_info["promoted"] = True
            stable_model_info["display_name"] = "Основная"
            artifact_store.save_json(MODEL_INFO_URI, stable_model_info)
            updated_registry = upsert_model_entry(
                registry,
                candidate_entry,
                promote=True,
            )
            artifact_store.save_json(MODEL_REGISTRY_URI, updated_registry)
            return payload

        for uri in [
            *payload["source_dataset_uris"],
            *payload["invalid_dataset_uris"],
        ]:
            artifact_store.move_uri(uri, _non_promoted_destination_uri(uri, version))

        rejected_entry = dict(candidate_entry)
        rejected_entry["status"] = "rejected"
        updated_registry = upsert_model_entry(
            registry,
            rejected_entry,
            promote=False,
        )
        artifact_store.save_json(MODEL_REGISTRY_URI, updated_registry)
        return payload

    @task
    def reload_model_if_promoted(payload: dict[str, object]) -> None:
        if not payload["promote"]:
            return
        status_code = _request(f"{INFERENCE_BASE_URL}/internal/reload-model", method="POST")
        if status_code != 200:
            raise RuntimeError(f"Unexpected reload status code: {status_code}")

    @task
    def validate_inference_health() -> None:
        status_code = _request(f"{INFERENCE_BASE_URL}/health")
        if status_code != 200:
            raise RuntimeError(f"Unexpected health status code: {status_code}")

    ready = wait_for_inference()
    prepared = prepare_candidate_dataset()
    trained = train_candidate_model(prepared)
    decision = compare_candidate_to_production(trained)
    applied = apply_promotion_decision(decision)
    ready >> prepared
    applied >> reload_model_if_promoted(applied) >> validate_inference_health()


churn_candidate_promotion_pipeline()
