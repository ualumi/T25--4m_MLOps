from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

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
AIRFLOW_PROMOTION_SCHEDULE = os.getenv("AIRFLOW_PROMOTION_SCHEDULE", "0 0 1 * *")
AIRFLOW_TRAIN_IMAGE = os.getenv(
    "AIRFLOW_TRAIN_IMAGE",
    os.getenv("INFERENCE_SERVICE_IMAGE", "mlops/inference-service:latest"),
)
AIRFLOW_K8S_NAMESPACE = os.getenv("AIRFLOW_K8S_NAMESPACE", "mlops-app")
AIRFLOW_K8S_SERVICE_ACCOUNT = os.getenv("AIRFLOW_K8S_SERVICE_ACCOUNT", "airflow-runner")
SOURCE_DATASETS_BUCKET = os.getenv("RETRAIN_SOURCE_DATASETS_BUCKET", "ml-artifacts")
SOURCE_DATASETS_PREFIX = os.getenv("RETRAIN_SOURCE_DATASETS_PREFIX", DEFAULT_RETRAIN_SOURCE_PREFIX)
NON_PROMOTED_PREFIX = os.getenv("RETRAIN_NON_PROMOTED_PREFIX", DEFAULT_RETRAIN_NON_PROMOTED_PREFIX)
REFERENCE_SCHEMA_PATH = os.getenv("RETRAIN_REFERENCE_SCHEMA_PATH", str(PROJECT_ROOT / "data" / "processed.csv"))
TEST_DATASET_PATH = os.getenv("RETRAIN_TEST_DATASET_PATH", str(PROJECT_ROOT / "data" / "test_data.csv"))
TOP_SHARE = float(os.getenv("RETRAIN_TOP_SHARE", "0.2"))
MODEL_URI = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
BASELINE_MODEL_URI = os.getenv("BASELINE_MODEL_URI", DEFAULT_BASELINE_MODEL_URI)
REPORT_URI = os.getenv("REPORT_URI", DEFAULT_REPORT_URI)
FEATURE_SCHEMA_URI = os.getenv("FEATURE_SCHEMA_URI", DEFAULT_FEATURE_SCHEMA_URI)
FEATURE_STATS_URI = os.getenv("FEATURE_STATS_URI", DEFAULT_FEATURE_STATS_URI)
MODEL_INFO_URI = os.getenv("MODEL_INFO_URI", DEFAULT_MODEL_INFO_URI)
MODEL_REGISTRY_URI = os.getenv("MODEL_REGISTRY_URI", DEFAULT_MODEL_REGISTRY_URI)

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
        sse_mode=os.getenv("S3_SSE_MODE"),
        sse_kms_key_id=os.getenv("S3_SSE_KMS_KEY_ID"),
    )


def _request(url: str, method: str = "GET") -> int:
    request = Request(url=url, method=method)
    with urlopen(request, timeout=10) as response:
        return response.getcode()


def _wait_for_inference() -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if _request(f"{INFERENCE_BASE_URL}/health") == 200:
                return
        except (HTTPError, URLError):
            pass
        time.sleep(5)
    raise RuntimeError("Сервис инференса не стал доступен до истечения таймаута.")


def _version_from_logical_date(logical_date: datetime) -> str:
    return logical_date.strftime("%Y%m%dT%H%M%SZ")


def _is_manual_run(context: dict[str, object]) -> bool:
    dag_run = context.get("dag_run")
    run_type = str(getattr(dag_run, "run_type", "")).lower()
    return run_type.endswith("manual")


def _artifact_uris(version: str) -> dict[str, str]:
    return build_versioned_artifact_uris(MODEL_REGISTRY_URI, version)


def _manifest_uri(version: str) -> str:
    bucket, key = parse_s3_uri(MODEL_REGISTRY_URI)
    prefix = str(Path(key).parent).replace("\\", "/").strip("/")
    if prefix == ".":
        return f"s3://{bucket}/runs/{version}/candidate_input_manifest.json"
    return f"s3://{bucket}/{prefix}/runs/{version}/candidate_input_manifest.json"


def _load_manifest(store: S3ArtifactStore, version: str) -> dict[str, list[str]] | None:
    try:
        payload = store.load_json(_manifest_uri(version))
    except FileNotFoundError:
        return None
    return {
        "source_dataset_uris": sorted({str(v) for v in payload.get("source_dataset_uris", [])}),
        "invalid_dataset_uris": sorted({str(v) for v in payload.get("invalid_dataset_uris", [])}),
    }


def _save_manifest(store: S3ArtifactStore, version: str, source_uris: list[str], invalid_uris: list[str]) -> None:
    store.save_json(
        _manifest_uri(version),
        {
            "version": version,
            "source_dataset_uris": sorted(set(source_uris)),
            "invalid_dataset_uris": sorted(set(invalid_uris)),
        },
    )


def _find_registry_entry(registry: dict[str, object], version: str) -> dict[str, object] | None:
    for entry in registry.get("models", []):
        if isinstance(entry, dict) and str(entry.get("version")) == version:
            return entry
    return None


def _validate_training_dataset(raw_bytes: bytes, required_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw_bytes))
    expected_columns = ["user_id", *required_columns, TARGET_COLUMN]
    if list(frame.columns) != expected_columns:
        raise ValueError("Колонки датасета не соответствуют обучающей схеме.")
    if frame.empty:
        raise ValueError("Обучающий датасет пуст.")
    values = set(frame[TARGET_COLUMN].dropna().astype(int).tolist())
    if not values.issubset({0, 1}):
        raise ValueError("Целевая колонка 'churn' должна содержать только значения 0/1.")
    return frame


def _noop(version: str, reason: str, invalid_uris: list[str] | None = None) -> dict[str, object]:
    return {
        "noop": True,
        "noop_reason": reason,
        "version": version,
        "candidate_train_path": "",
        "source_dataset_uris": [],
        "invalid_dataset_uris": invalid_uris or [],
        "artifact_uris": _artifact_uris(version),
    }


def _move_if_exists(store: S3ArtifactStore, source_uri: str, destination_uri: str) -> None:
    try:
        store.download_bytes(source_uri)
    except FileNotFoundError:
        return
    store.move_uri(source_uri, destination_uri)


def _non_promoted_destination_uri(source_uri: str, version: str) -> str:
    bucket, key = parse_s3_uri(source_uri)
    return f"s3://{bucket}/{NON_PROMOTED_PREFIX.strip('/')}/{version}/{Path(key).name}"


def _candidate_train_uri(version: str) -> str:
    bucket, key = parse_s3_uri(MODEL_REGISTRY_URI)
    prefix = str(Path(key).parent).replace("\\", "/").strip("/")
    if prefix == ".":
        return f"s3://{bucket}/runs/{version}/candidate-train.csv"
    return f"s3://{bucket}/{prefix}/runs/{version}/candidate-train.csv"


def _train_environment() -> dict[str, str]:
    keys = (
        "S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "S3_SSE_MODE",
        "S3_SSE_KMS_KEY_ID",
    )
    return {k: v for k in keys if (v := os.getenv(k)) is not None}


def _train_command_template() -> str:
    return f"""
import json
import os
import sys
import tempfile

from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore
from src.training.train import main

payload = json.loads(r'''{{{{ ti.xcom_pull(task_ids='prepare_candidate_dataset') | tojson }}}}''')
if payload.get("noop"):
    print(f"Пропуск запуска (no-op): {{payload.get('noop_reason', '')}}")
else:
    train_path = payload["candidate_train_path"]
    if train_path.startswith("s3://"):
        store = S3ArtifactStore(
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            sse_mode=os.getenv("S3_SSE_MODE"),
            sse_kms_key_id=os.getenv("S3_SSE_KMS_KEY_ID"),
        )
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as file_obj:
            file_obj.write(store.download_bytes(train_path))
            train_path = file_obj.name

    args = [
        "train",
        "--train-path",
        train_path,
        "--test-path",
        "/app/data/test_data.csv",
        "--production-model-uri",
        payload["artifact_uris"]["production_model_uri"],
        "--baseline-model-uri",
        payload["artifact_uris"]["baseline_model_uri"],
        "--report-uri",
        payload["artifact_uris"]["report_uri"],
        "--feature-schema-uri",
        payload["artifact_uris"]["feature_schema_uri"],
        "--feature-stats-uri",
        payload["artifact_uris"]["feature_stats_uri"],
        "--model-info-uri",
        payload["artifact_uris"]["model_info_uri"],
        "--model-registry-uri",
        "{MODEL_REGISTRY_URI}",
        "--top-share",
        "{TOP_SHARE}",
        "--version",
        payload["version"],
    ]
    for uri in payload.get("source_dataset_uris", []):
        args.extend(["--source-dataset-uri", uri])

    sys.argv = args
    main()
"""


@dag(
    dag_id="churn_candidate_promotion_pipeline",
    description="Idempotent candidate promotion DAG.",
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    schedule=AIRFLOW_PROMOTION_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "training", "promotion", "churn"],
)
def churn_candidate_promotion_pipeline():
    @task
    def wait_for_inference() -> None:
        _wait_for_inference()

    @task
    def prepare_candidate_dataset() -> dict[str, object]:
        context = get_current_context()
        store = _artifact_store()
        registry = load_model_registry(store, MODEL_REGISTRY_URI)
        manual_run = _is_manual_run(context)
        current_version = registry.get("current_version")

        if manual_run and current_version:
            version = str(current_version)
        else:
            version = _version_from_logical_date(context["logical_date"])

        existing = _find_registry_entry(registry, version)
        if existing is not None and str(existing.get("status")) in {"production", "rejected"}:
            return _noop(version, f"Версия {version} уже финализирована.")

        required_columns = get_feature_columns(load_dataset(REFERENCE_SCHEMA_PATH))
        processed = {
            str(uri)
            for entry in registry.get("models", [])
            if str(entry.get("version")) != version
            for uri in entry.get("source_dataset_uris", [])
        }

        manifest = _load_manifest(store, version)
        valid_uris: list[str] = []
        invalid_uris: list[str] = []
        frames: list[pd.DataFrame] = []

        if manifest is None:
            for uri in store.list_uris(SOURCE_DATASETS_BUCKET, SOURCE_DATASETS_PREFIX):
                if (not uri.lower().endswith(".csv")) or (uri in processed):
                    continue
                try:
                    frame = _validate_training_dataset(store.download_bytes(uri), required_columns)
                except Exception:
                    invalid_uris.append(uri)
                    continue
                valid_uris.append(uri)
                frames.append(frame)
            _save_manifest(store, version, valid_uris, invalid_uris)
        else:
            valid_uris = manifest["source_dataset_uris"]
            invalid_uris = manifest["invalid_dataset_uris"]
            for uri in valid_uris:
                frames.append(_validate_training_dataset(store.download_bytes(uri), required_columns))

        if not frames:
            return _noop(version, "Не найдено новых валидных датасетов для обучения.", invalid_uris)

        merged = pd.concat(frames, ignore_index=True)
        if len(set(merged[TARGET_COLUMN].astype(int).tolist())) < 2:
            return _noop(version, "Объединенный датасет не содержит оба класса.", invalid_uris)

        candidate_train_uri = _candidate_train_uri(version)
        payload = merged.to_csv(index=False).encode("utf-8")
        store.upload_bytes(candidate_train_uri, payload, content_type="text/csv")

        return {
            "noop": False,
            "noop_reason": "",
            "version": version,
            "candidate_train_path": candidate_train_uri,
            "source_dataset_uris": sorted(set(valid_uris)),
            "invalid_dataset_uris": sorted(set(invalid_uris)),
            "artifact_uris": _artifact_uris(version),
        }

    @task
    def after_training(payload: dict[str, object], _train_finished: object) -> dict[str, object]:
        del _train_finished
        return payload

    @task
    def compare_candidate_to_production(payload: dict[str, object]) -> dict[str, object]:
        if payload["noop"]:
            return {**payload, "promote": False}

        candidate_metrics = evaluate_saved_model(
            model_uri=payload["artifact_uris"]["production_model_uri"],
            dataset_path=TEST_DATASET_PATH,
            top_share=TOP_SHARE,
        )
        registry = load_model_registry(_artifact_store(), MODEL_REGISTRY_URI)
        current_version = registry.get("current_version")
        if current_version is None:
            return {**payload, "promote": True, "candidate_metrics": candidate_metrics}

        production_metrics = evaluate_saved_model(
            model_uri=MODEL_URI,
            dataset_path=TEST_DATASET_PATH,
            top_share=TOP_SHARE,
        )
        return {
            **payload,
            "promote": candidate_metrics["roc_auc"] > production_metrics["roc_auc"],
            "candidate_metrics": candidate_metrics,
            "production_metrics": production_metrics,
        }

    @task
    def apply_promotion_decision(payload: dict[str, object]) -> dict[str, object]:
        if payload["noop"]:
            return payload

        store = _artifact_store()
        registry = load_model_registry(store, MODEL_REGISTRY_URI)
        version = str(payload["version"])
        candidate_entry = _find_registry_entry(registry, version)
        if candidate_entry is None:
            raise RuntimeError(f"В реестре моделей нет записи для версии={version}.")

        if payload["promote"]:
            store.copy_uri(payload["artifact_uris"]["production_model_uri"], MODEL_URI)
            store.copy_uri(payload["artifact_uris"]["baseline_model_uri"], BASELINE_MODEL_URI)
            store.copy_uri(payload["artifact_uris"]["report_uri"], REPORT_URI)
            store.copy_uri(payload["artifact_uris"]["feature_schema_uri"], FEATURE_SCHEMA_URI)
            store.copy_uri(payload["artifact_uris"]["feature_stats_uri"], FEATURE_STATS_URI)
            model_info = store.load_json(payload["artifact_uris"]["model_info_uri"])
            model_info["promoted"] = True
            model_info["display_name"] = "Основная"
            store.save_json(MODEL_INFO_URI, model_info)
            store.save_json(MODEL_REGISTRY_URI, upsert_model_entry(registry, candidate_entry, promote=True))
            return payload

        for uri in [*payload["source_dataset_uris"], *payload["invalid_dataset_uris"]]:
            _move_if_exists(store, str(uri), _non_promoted_destination_uri(str(uri), version))

        rejected_entry = dict(candidate_entry)
        rejected_entry["status"] = "rejected"
        store.save_json(MODEL_REGISTRY_URI, upsert_model_entry(registry, rejected_entry, promote=False))
        return payload

    @task
    def reload_model_if_promoted(payload: dict[str, object]) -> None:
        if payload["promote"]:
            status_code = _request(f"{INFERENCE_BASE_URL}/internal/reload-model", method="POST")
            if status_code != 200:
                raise RuntimeError(f"Неожиданный код ответа при перезагрузке модели: {status_code}")

    @task
    def validate_inference_health() -> None:
        status_code = _request(f"{INFERENCE_BASE_URL}/health")
        if status_code != 200:
            raise RuntimeError(f"Неожиданный код ответа health-check: {status_code}")

    ready = wait_for_inference()
    prepared = prepare_candidate_dataset()
    train_task = KubernetesPodOperator(
        task_id="train_candidate_model",
        name="train-candidate-model",
        image=AIRFLOW_TRAIN_IMAGE,
        cmds=["python"],
        arguments=["-c", _train_command_template()],
        namespace=AIRFLOW_K8S_NAMESPACE,
        service_account_name=AIRFLOW_K8S_SERVICE_ACCOUNT,
        env_vars=_train_environment(),
        in_cluster=True,
        get_logs=True,
        is_delete_operator_pod=True,
        do_xcom_push=False,
    )
    trained = after_training(prepared, train_task.output)
    decision = compare_candidate_to_production(trained)
    applied = apply_promotion_decision(decision)

    ready >> prepared >> train_task
    applied >> reload_model_if_promoted(applied) >> validate_inference_health()


churn_candidate_promotion_pipeline()
