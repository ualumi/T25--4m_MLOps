"""DAG: датасеты из training/ → обучение LightGBM → candidate_to_production → сравнение с production → промоушен.

Идемпотентность candidate в S3 (перезапись того же префикса):
- JSON: ``{"candidate_key": "my-label"}`` → ``.../candidate_to_production/my-label/`` (явный ключ)
- Иначе ключ = **logical date** запуска (дата в UI триггера / интервал DAG): ``.../candidate_to_production/<YYYY-MM-DD>/``
  Ручной триггер с датой «2 месяца назад» пишет в папку той даты; повтор с той же датой перезаписывает тот же candidate.
- Объединённый датасет: ``.../candidate_to_production/<key>/merged.csv`` (рядом с артефактами кандидата).
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from airflow.decorators import dag, task
from airflow.models import BaseOperator
from airflow.operators.python import get_current_context
from airflow.providers.docker.operators.docker import DockerOperator

PROJECT_ROOT = Path("/opt/project")
INFERENCE_SERVICE_ROOT = PROJECT_ROOT / "services" / "inference-service"
for candidate_path in (PROJECT_ROOT, INFERENCE_SERVICE_ROOT):
    if str(candidate_path) not in sys.path:
        sys.path.append(str(candidate_path))

from config.config import normalize_retrain_source_prefix  # noqa: E402
from config.s3_layout import (  # noqa: E402
    DEFAULT_BASELINE_MODEL_URI,
    DEFAULT_CANDIDATE_TO_PRODUCTION_PREFIX,
    DEFAULT_FEATURE_SCHEMA_URI,
    DEFAULT_FEATURE_STATS_URI,
    DEFAULT_MODEL_INFO_URI,
    DEFAULT_MODEL_REGISTRY_URI,
    DEFAULT_MODEL_URI,
    DEFAULT_REPORT_URI,
    DEFAULT_RETRAIN_SOURCE_PREFIX,
    DEFAULT_VERSION_ARCHIVE_PREFIX,
)
from src.infrastructure.storage.s3_artifact_store import (  # noqa: E402
    S3ArtifactStore,
    parse_s3_uri,
)
from src.training.data import TARGET_COLUMN, get_feature_columns, load_dataset  # noqa: E402
from src.training.evaluate import evaluate_saved_model  # noqa: E402
from src.training.model_registry import load_model_registry, upsert_model_entry  # noqa: E402

INFERENCE_BASE_URL = os.getenv("AIRFLOW_INFERENCE_BASE_URL", "http://inference-service:8001")
AIRFLOW_TRAIN_IMAGE = os.getenv(
    "AIRFLOW_TRAIN_IMAGE",
    os.getenv("INFERENCE_SERVICE_IMAGE", "mlops/inference-service:latest"),
)
SOURCE_DATASETS_BUCKET = os.getenv("RETRAIN_SOURCE_DATASETS_BUCKET", "ml-artifacts")
SOURCE_DATASETS_PREFIX = normalize_retrain_source_prefix(
    os.getenv("RETRAIN_SOURCE_DATASETS_PREFIX", DEFAULT_RETRAIN_SOURCE_PREFIX)
)
REFERENCE_SCHEMA_PATH = os.getenv("RETRAIN_REFERENCE_SCHEMA_PATH", str(PROJECT_ROOT / "data" / "processed.csv"))
TEST_DATASET_PATH = os.getenv("RETRAIN_TEST_DATASET_PATH", str(PROJECT_ROOT / "data" / "test_data.csv"))
TOP_SHARE = float(os.getenv("RETRAIN_TOP_SHARE", "0.2"))
MIN_ROC_DELTA = float(os.getenv("PROMOTION_MIN_ROC_DELTA", "0.005"))

MODEL_URI = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
BASELINE_MODEL_URI = os.getenv("BASELINE_MODEL_URI", DEFAULT_BASELINE_MODEL_URI)
REPORT_URI = os.getenv("REPORT_URI", DEFAULT_REPORT_URI)
FEATURE_SCHEMA_URI = os.getenv("FEATURE_SCHEMA_URI", DEFAULT_FEATURE_SCHEMA_URI)
FEATURE_STATS_URI = os.getenv("FEATURE_STATS_URI", DEFAULT_FEATURE_STATS_URI)
MODEL_INFO_URI = os.getenv("MODEL_INFO_URI", DEFAULT_MODEL_INFO_URI)
MODEL_REGISTRY_URI = os.getenv("MODEL_REGISTRY_URI", DEFAULT_MODEL_REGISTRY_URI)

AIRFLOW_SCHEDULE = os.getenv("AIRFLOW_TRAINING_TO_PRODUCTION_SCHEDULE", "0 3 1 * *")

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def make_train_operator(
    *,
    task_id: str,
    image: str,
    arguments_template: str,
    env_vars: dict[str, str],
) -> BaseOperator:
    return DockerOperator(
        task_id=task_id,
        docker_url=os.getenv("AIRFLOW_DOCKER_URL", "unix://var/run/docker.sock"),
        image=image,
        command=["python", "-c", arguments_template],
        environment=env_vars,
        network_mode=os.getenv("AIRFLOW_DOCKER_NETWORK", "mlops_network"),
        api_version="auto",
        auto_remove="force",
        tty=False,
        mount_tmp_dir=False,
    )


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
    with urlopen(request, timeout=30) as response:
        return response.getcode()


def _wait_for_inference() -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if _request(f"{INFERENCE_BASE_URL}/health") == 200:
                return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(5)
    raise RuntimeError("Сервис инференса не ответил по /health до таймаута.")


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


def _noop(reason: str) -> dict[str, Any]:
    return {"noop": True, "noop_reason": reason}


def _sanitize_candidate_key(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip()).strip("._-")
    return cleaned[:120] if cleaned else "default"


def _candidate_idempotency_key(context: dict[str, Any]) -> str:
    """Сегмент пути S3: тот же logical date (или candidate_key) → та же папка candidate."""
    dag_run = context.get("dag_run")
    conf: dict[str, Any] = {}
    if dag_run is not None:
        raw_conf = getattr(dag_run, "conf", None) or {}
        if isinstance(raw_conf, dict):
            conf = raw_conf
    explicit = conf.get("candidate_key") or conf.get("candidate_suffix")
    if explicit is not None and str(explicit).strip():
        return _sanitize_candidate_key(str(explicit).strip())
    logical = context["logical_date"]
    return logical.strftime("%Y-%m-%d")


def _safe_archive_label(version: object | None) -> str:
    raw = str(version or "legacy").strip()[:180]
    return raw.replace("/", "-").replace(" ", "_")


def _find_registry_entry(registry: dict[str, object], version: str) -> dict[str, object] | None:
    for entry in registry.get("models", []):
        if isinstance(entry, dict) and str(entry.get("version")) == version:
            return entry
    return None


def _train_environment() -> dict[str, str]:
    keys = (
        "S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "S3_SSE_MODE",
        "S3_SSE_KMS_KEY_ID",
        "MODEL_URI",
        "BASELINE_MODEL_URI",
        "REPORT_URI",
        "FEATURE_SCHEMA_URI",
        "FEATURE_STATS_URI",
        "MODEL_INFO_URI",
        "MODEL_REGISTRY_URI",
    )
    out: dict[str, str] = {}
    for key in keys:
        val = os.getenv(key)
        if val is not None:
            out[key] = val
    return out


def _train_command_template() -> str:
    return """
import json
import os
import sys
import tempfile

from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore
from src.training.train import main

_raw = r'''{{ ti.xcom_pull(task_ids='03_merge_and_validate') | tojson }}'''
try:
    _parsed = json.loads(_raw)
except json.JSONDecodeError:
    _parsed = None
payload = _parsed if isinstance(_parsed, dict) else {}
if not payload:
    raise RuntimeError(
        "XCom от задачи 03_merge_and_validate пуст или не dict — проверь task_id и upstream."
    )
if payload.get("noop"):
    print("Пропуск обучения:", payload.get("noop_reason", ""))
    sys.exit(0)

store = S3ArtifactStore(
    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    sse_mode=os.getenv("S3_SSE_MODE"),
    sse_kms_key_id=os.getenv("S3_SSE_KMS_KEY_ID"),
)
mp = payload["merged_train_uri"]
if mp.startswith("s3://"):
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
        f.write(store.download_bytes(mp))
        train_path = f.name
else:
    train_path = mp

sys.argv = [
    "train",
    "--train-path",
    train_path,
    "--test-path",
    "/app/data/test_data.csv",
    "--artifact-output-prefix",
    payload["candidate_prefix"],
    "--registry-read-uri",
    payload["model_registry_uri"],
    "--model-registry-uri",
    payload["model_registry_uri"],
    "--production-model-uri",
    os.environ["MODEL_URI"],
    "--baseline-model-uri",
    os.environ["BASELINE_MODEL_URI"],
    "--report-uri",
    os.environ["REPORT_URI"],
    "--feature-schema-uri",
    os.environ["FEATURE_SCHEMA_URI"],
    "--feature-stats-uri",
    os.environ["FEATURE_STATS_URI"],
    "--model-info-uri",
    os.environ["MODEL_INFO_URI"],
    "--top-share",
    "__TOP_SHARE__",
    "--version",
    payload["version"],
]
for uri in payload.get("source_dataset_uris", []):
    sys.argv.extend(["--source-dataset-uri", uri])
main()
""".replace(
        "__TOP_SHARE__", str(TOP_SHARE)
    )


@dag(
    dag_id="training_to_production_pipeline",
    description="training/ → candidate_to_production → ROC-AUC vs production (+0.005) → промоушен и архив.",
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    schedule=AIRFLOW_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "training", "promotion", "churn"],
)
def training_to_production_pipeline():
    @task(task_id="01_inference_health")
    def task_01_inference_health() -> None:
        _wait_for_inference()

    @task(task_id="02_validate_training_csvs")
    def task_02_validate_sources() -> dict[str, Any]:
        store = _artifact_store()
        required_columns = get_feature_columns(load_dataset(REFERENCE_SCHEMA_PATH))
        valid_uris: list[str] = []
        invalid_uris: list[str] = []
        first_invalid_error: str | None = None
        for uri in store.list_uris(SOURCE_DATASETS_BUCKET, SOURCE_DATASETS_PREFIX):
            if not uri.lower().endswith(".csv"):
                continue
            try:
                _validate_training_dataset(store.download_bytes(uri), required_columns)
            except Exception as exc:
                invalid_uris.append(uri)
                if first_invalid_error is None:
                    first_invalid_error = str(exc).strip()[:400]
                continue
            valid_uris.append(uri)
        if not valid_uris:
            bucket = SOURCE_DATASETS_BUCKET
            prefix = SOURCE_DATASETS_PREFIX.strip("/")
            endpoint = os.getenv("S3_ENDPOINT_URL", "")
            if not invalid_uris:
                noop_reason = (
                    f"В s3://{bucket}/{prefix}/ нет объектов с суффиксом .csv "
                    f"(или список пуст — проверьте S3_ENDPOINT_URL={endpoint!r} и те же бакет/префикс, что в Gateway)."
                )
            else:
                detail = (
                    f"Первый отказ: {first_invalid_error}"
                    if first_invalid_error
                    else "см. invalid_uris в XCom."
                )
                noop_reason = (
                    f"Найдено {len(invalid_uris)} CSV, ни один не соответствует схеме обучения "
                    f"(колонки и порядок как в эталоне REFERENCE_SCHEMA / processed.csv; churn только 0/1). "
                    f"{detail}"
                )
            return {
                **_noop(noop_reason),
                "valid_uris": [],
                "invalid_uris": sorted(set(invalid_uris)),
            }
        return {
            "noop": False,
            "valid_uris": sorted(set(valid_uris)),
            "invalid_uris": sorted(set(invalid_uris)),
        }

    @task(task_id="03_merge_and_validate")
    def task_03_merge(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("noop"):
            return payload

        store = _artifact_store()
        required_columns = get_feature_columns(load_dataset(REFERENCE_SCHEMA_PATH))
        frames: list[pd.DataFrame] = []
        for uri in payload["valid_uris"]:
            frames.append(_validate_training_dataset(store.download_bytes(uri), required_columns))
        merged = pd.concat(frames, ignore_index=True)
        if len(set(merged[TARGET_COLUMN].astype(int).tolist())) < 2:
            return {**_noop("В объединённом датасете один класс churn."), **payload}

        context = get_current_context()
        logical: datetime = context["logical_date"]
        bucket = SOURCE_DATASETS_BUCKET
        idem = _candidate_idempotency_key(context)
        candidate_rel = f"{DEFAULT_CANDIDATE_TO_PRODUCTION_PREFIX.strip('/')}/{idem}"
        merge_key = f"{candidate_rel}/merged.csv"
        merged_uri = f"s3://{bucket}/{merge_key}"
        store.upload_bytes(merged_uri, merged.to_csv(index=False).encode("utf-8"), content_type="text/csv")

        candidate_prefix = f"s3://{bucket}/{candidate_rel}"
        version = logical.strftime("%Y%m%dT%H%M%SZ")

        return {
            "noop": False,
            "merged_train_uri": merged_uri,
            "candidate_prefix": candidate_prefix,
            "candidate_idempotency_key": idem,
            "version": version,
            "model_registry_uri": MODEL_REGISTRY_URI,
            "source_dataset_uris": payload["valid_uris"],
            "invalid_uris": payload.get("invalid_uris", []),
        }

    @task(task_id="04_pass_through_after_train")
    def task_04_after_train(merge_payload: dict[str, Any], _train_done: Any) -> dict[str, Any]:
        del _train_done
        return merge_payload

    @task(task_id="05_verify_candidate_artifacts")
    def task_05_verify_candidate(merge_payload: dict[str, Any]) -> dict[str, Any]:
        if merge_payload.get("noop"):
            return merge_payload
        store = _artifact_store()
        base = merge_payload["candidate_prefix"].rstrip("/")
        for name in (
            "model.joblib",
            "baseline.joblib",
            "training_metrics.json",
            "feature_schema.json",
            "feature_stats.json",
            "model_info.json",
            "model_registry.json",
        ):
            store.download_bytes(f"{base}/{name}")
        metrics = store.load_json(f"{base}/training_metrics.json")
        roc = metrics.get("mvp_lightgbm", {}).get("roc_auc")
        if roc is None:
            raise RuntimeError("В training_metrics.json нет mvp_lightgbm.roc_auc.")
        return merge_payload

    @task(task_id="06_compare_roc_auc")
    def task_06_compare(merge_payload: dict[str, Any]) -> dict[str, Any]:
        if merge_payload.get("noop"):
            return {**merge_payload, "promote": False}

        store = _artifact_store()
        cand_model = f"{merge_payload['candidate_prefix'].rstrip('/')}/model.joblib"
        candidate_metrics = evaluate_saved_model(
            model_uri=cand_model,
            dataset_path=TEST_DATASET_PATH,
            top_share=TOP_SHARE,
        )
        c_roc = float(candidate_metrics["roc_auc"])

        try:
            store.download_bytes(MODEL_URI)
        except FileNotFoundError:
            return {
                **merge_payload,
                "promote": True,
                "candidate_metrics": candidate_metrics,
                "production_metrics": None,
                "roc_delta": None,
                "reason": "Нет production-модели в S3 — принимаем кандидата.",
            }

        production_metrics = evaluate_saved_model(
            model_uri=MODEL_URI,
            dataset_path=TEST_DATASET_PATH,
            top_share=TOP_SHARE,
        )
        p_roc = float(production_metrics["roc_auc"])
        delta = c_roc - p_roc
        promote = delta >= MIN_ROC_DELTA
        return {
            **merge_payload,
            "promote": promote,
            "candidate_metrics": candidate_metrics,
            "production_metrics": production_metrics,
            "roc_delta": delta,
            "reason": f"candidate roc_auc={c_roc:.4f}, production={p_roc:.4f}, delta={delta:.4f}, min_delta={MIN_ROC_DELTA}",
        }

    @task(task_id="07_archive_old_production")
    def task_07_archive(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("noop") or not payload.get("promote"):
            return payload

        store = _artifact_store()
        registry = load_model_registry(store, MODEL_REGISTRY_URI)
        prev_ver = registry.get("current_version")
        label = _safe_archive_label(prev_ver)
        bucket, _ = parse_s3_uri(MODEL_URI)
        archive_base = f"s3://{bucket}/{DEFAULT_VERSION_ARCHIVE_PREFIX.strip('/')}/{label}"

        for src in (
            MODEL_URI,
            BASELINE_MODEL_URI,
            REPORT_URI,
            FEATURE_SCHEMA_URI,
            FEATURE_STATS_URI,
            MODEL_INFO_URI,
            MODEL_REGISTRY_URI,
        ):
            try:
                data = store.download_bytes(src)
            except FileNotFoundError:
                continue
            _, key = parse_s3_uri(src)
            name = Path(key).name
            store.upload_bytes(f"{archive_base}/{name}", data)

        return {**payload, "archive_prefix": archive_base, "archived_from_version": prev_ver}

    @task(task_id="08_promote_candidate_to_production")
    def task_08_promote(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("noop") or not payload.get("promote"):
            return payload

        store = _artifact_store()
        cand = payload["candidate_prefix"].rstrip("/")
        store.copy_uri(f"{cand}/model.joblib", MODEL_URI)
        store.copy_uri(f"{cand}/baseline.joblib", BASELINE_MODEL_URI)
        store.copy_uri(f"{cand}/training_metrics.json", REPORT_URI)
        store.copy_uri(f"{cand}/feature_schema.json", FEATURE_SCHEMA_URI)
        store.copy_uri(f"{cand}/feature_stats.json", FEATURE_STATS_URI)

        c_info = store.load_json(f"{cand}/model_info.json")
        c_info["promoted"] = True
        c_info["display_name"] = "Основная"
        store.save_json(MODEL_INFO_URI, c_info)

        cand_reg = store.load_json(f"{cand}/model_registry.json")
        version = str(c_info.get("version", payload.get("version", "")))
        entry = _find_registry_entry(cand_reg, version)
        if entry is None:
            raise RuntimeError(f"Нет записи версии {version} в candidate model_registry.json")
        prod_reg = load_model_registry(store, MODEL_REGISTRY_URI)
        store.save_json(MODEL_REGISTRY_URI, upsert_model_entry(prod_reg, entry, promote=True))

        return payload

    @task(task_id="09_verify_production_updated")
    def task_09_verify_production(payload: dict[str, Any]) -> None:
        if payload.get("noop") or not payload.get("promote"):
            return
        store = _artifact_store()
        store.download_bytes(MODEL_URI)
        info = store.load_json(MODEL_INFO_URI)
        if not info.get("promoted"):
            raise RuntimeError("model_info в production не помечен как promoted.")

    @task(task_id="10_verify_archive_exists")
    def task_10_verify_archive(payload: dict[str, Any]) -> None:
        if payload.get("noop") or not payload.get("promote"):
            return
        ap = payload.get("archive_prefix")
        if not ap:
            raise RuntimeError("Нет archive_prefix в XCom.")
        store = _artifact_store()
        store.download_bytes(f"{ap.rstrip('/')}/model.joblib")

    @task(task_id="11_reload_inference_model")
    def task_11_reload(
        payload: dict[str, Any],
        _verified_prod: None,
        _verified_arch: None,
    ) -> None:
        del _verified_prod, _verified_arch
        if payload.get("noop") or not payload.get("promote"):
            return
        code = _request(f"{INFERENCE_BASE_URL}/internal/reload-model", method="POST")
        if code != 200:
            raise RuntimeError(f"reload-model вернул {code}")

    t01 = task_01_inference_health()
    t02 = task_02_validate_sources()
    t03 = task_03_merge(t02)

    train_task = make_train_operator(
        task_id="train_lightgbm_save_candidate",
        image=AIRFLOW_TRAIN_IMAGE,
        arguments_template=_train_command_template(),
        env_vars=_train_environment(),
    )

    t04 = task_04_after_train(t03, train_task.output)
    t05 = task_05_verify_candidate(t04)
    t06 = task_06_compare(t05)
    t07 = task_07_archive(t06)
    t08 = task_08_promote(t07)
    t09 = task_09_verify_production(t08)
    t10 = task_10_verify_archive(t08)
    t11 = task_11_reload(t08, t09, t10)

    t01 >> t02 >> t03 >> train_task
    t04 >> t05 >> t06 >> t07 >> t08


training_to_production_pipeline()
