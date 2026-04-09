from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.providers.docker.operators.docker import DockerOperator

PROJECT_ROOT = Path("/opt/project")
INFERENCE_SERVICE_ROOT = PROJECT_ROOT / "services" / "inference-service"
for candidate_path in (PROJECT_ROOT, INFERENCE_SERVICE_ROOT):
    if str(candidate_path) not in sys.path:
        sys.path.append(str(candidate_path))

from config.s3_layout import DEFAULT_MODEL_REGISTRY_URI  # noqa: E402
from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore  # noqa: E402
from src.training.model_registry import load_model_registry  # noqa: E402

INFERENCE_BASE_URL = os.getenv("AIRFLOW_INFERENCE_BASE_URL", "http://inference-service:8001")
AIRFLOW_RETRAIN_SCHEDULE = os.getenv("AIRFLOW_RETRAIN_SCHEDULE", "0 2 * * *")
AIRFLOW_TRAIN_IMAGE = os.getenv(
    "AIRFLOW_TRAIN_IMAGE",
    os.getenv("INFERENCE_SERVICE_IMAGE", "mlops/inference-service:latest"),
)
AIRFLOW_DOCKER_NETWORK = os.getenv("AIRFLOW_DOCKER_NETWORK", "mlops_network")
AIRFLOW_DOCKER_HOST = os.getenv("AIRFLOW_DOCKER_HOST", "unix://var/run/docker.sock")
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
    )


def _version_from_logical_date(logical_date: datetime) -> str:
    return logical_date.strftime("%Y%m%dT%H%M%SZ")


def _is_manual_run(context: dict[str, object]) -> bool:
    dag_run = context.get("dag_run")
    run_type = str(getattr(dag_run, "run_type", "")).lower()
    return run_type.endswith("manual")


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
        "MODEL_URI",
        "BASELINE_MODEL_URI",
        "REPORT_URI",
        "FEATURE_SCHEMA_URI",
        "FEATURE_STATS_URI",
        "MODEL_INFO_URI",
        "MODEL_REGISTRY_URI",
    )
    return {k: v for k in keys if (v := os.getenv(k)) is not None}


def _train_command_template() -> str:
    return (
        "{% set p = ti.xcom_pull(task_ids='prepare_retraining_run') %}"
        "{% if not p['should_train'] %}"
        "print('Пропуск запуска (no-op): {{ p['reason'] }}')"
        "{% else %}"
        "import sys, os\n"
        "from src.training.train import main\n"
        "sys.argv = [\n"
        "  'train',\n"
        "  '--train-path', '/app/data/processed.csv',\n"
        "  '--test-path', '/app/data/test_data.csv',\n"
        "  '--production-model-uri', os.getenv('MODEL_URI', ''),\n"
        "  '--baseline-model-uri', os.getenv('BASELINE_MODEL_URI', ''),\n"
        "  '--report-uri', os.getenv('REPORT_URI', ''),\n"
        "  '--feature-schema-uri', os.getenv('FEATURE_SCHEMA_URI', ''),\n"
        "  '--feature-stats-uri', os.getenv('FEATURE_STATS_URI', ''),\n"
        "  '--model-info-uri', os.getenv('MODEL_INFO_URI', ''),\n"
        "  '--model-registry-uri', os.getenv('MODEL_REGISTRY_URI', ''),\n"
        "  '--version', '{{ p['version'] }}',\n"
        "  '--promote',\n"
        "]\n"
        "main()\n"
        "{% endif %}"
    )


@dag(
    dag_id="churn_retraining_pipeline",
    description="Идемпотентный DAG для переобучения.",
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    schedule=AIRFLOW_RETRAIN_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "training", "churn"],
)
def churn_retraining_pipeline():
    @task
    def wait_for_inference() -> None:
        _wait_for_inference()

    @task
    def prepare_retraining_run() -> dict[str, object]:
        context = get_current_context()
        registry = load_model_registry(_artifact_store(), MODEL_REGISTRY_URI)
        manual_run = _is_manual_run(context)
        current_version = registry.get("current_version")

        if manual_run and current_version:
            return {
                "version": str(current_version),
                "should_train": True,
                "reason": "Ручной запуск: перезаписываем текущую прод-версию.",
            }

        version = _version_from_logical_date(context["logical_date"])
        existing = _find_registry_entry(registry, version)
        if existing is not None:
            return {
                "version": version,
                "should_train": False,
                "reason": f"Версия {version} уже существует со статусом {existing.get('status')}.",
            }
        return {"version": version, "should_train": True, "reason": ""}

    @task
    def reload_if_trained(payload: dict[str, object], _train_finished: object) -> None:
        del _train_finished
        if payload["should_train"]:
            status_code = _request(f"{INFERENCE_BASE_URL}/internal/reload-model", method="POST")
            if status_code != 200:
                raise RuntimeError(f"Неожиданный код ответа при перезагрузке модели: {status_code}")

    @task
    def validate_inference_health() -> None:
        status_code = _request(f"{INFERENCE_BASE_URL}/health")
        if status_code != 200:
            raise RuntimeError(f"Неожиданный код ответа health-check: {status_code}")

    ready = wait_for_inference()
    prepared = prepare_retraining_run()
    train_task = DockerOperator(
        task_id="train_model",
        image=AIRFLOW_TRAIN_IMAGE,
        entrypoint=["python"],
        command=["-c", _train_command_template()],
        docker_url=AIRFLOW_DOCKER_HOST,
        network_mode=AIRFLOW_DOCKER_NETWORK,
        environment=_train_environment(),
        working_dir="/app",
        auto_remove="success",
        mount_tmp_dir=False,
        force_pull=False,
        do_xcom_push=True,
    )

    ready >> prepared >> train_task
    reload_if_trained(prepared, train_task.output) >> validate_inference_health()


churn_retraining_pipeline()
