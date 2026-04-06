from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

INFERENCE_BASE_URL = os.getenv("AIRFLOW_INFERENCE_BASE_URL", "http://inference-service:8001")
AIRFLOW_TRAIN_IMAGE = os.getenv(
    "AIRFLOW_TRAIN_IMAGE",
    os.getenv("INFERENCE_SERVICE_IMAGE", "mlops/inference-service:latest"),
)
AIRFLOW_DOCKER_NETWORK = os.getenv("AIRFLOW_DOCKER_NETWORK", "mlops_network")
AIRFLOW_DOCKER_HOST = os.getenv("AIRFLOW_DOCKER_HOST", "unix://var/run/docker.sock")

_TRAIN_ENV_KEYS = (
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


def _train_container_environment() -> dict[str, str]:
    return {k: v for k in _TRAIN_ENV_KEYS if (v := os.getenv(k)) is not None}


def _train_command() -> list[str]:
    return [
        "python",
        "-m",
        "src.training.train",
        "--train-path",
        "/app/data/processed.csv",
        "--test-path",
        "/app/data/test_data.csv",
        "--production-model-uri",
        os.getenv("MODEL_URI", ""),
        "--baseline-model-uri",
        os.getenv("BASELINE_MODEL_URI", ""),
        "--report-uri",
        os.getenv("REPORT_URI", ""),
        "--feature-schema-uri",
        os.getenv("FEATURE_SCHEMA_URI", ""),
        "--feature-stats-uri",
        os.getenv("FEATURE_STATS_URI", ""),
        "--model-info-uri",
        os.getenv("MODEL_INFO_URI", ""),
        "--model-registry-uri",
        os.getenv("MODEL_REGISTRY_URI", ""),
        "--promote",
    ]


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _request(url: str, method: str = "GET") -> int:
    request = Request(url=url, method=method)
    with urlopen(request, timeout=10) as response:
        return response.getcode()


def wait_for_inference() -> None:
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


def reload_model() -> None:
    status_code = _request(f"{INFERENCE_BASE_URL}/internal/reload-model", method="POST")
    if status_code != 200:
        raise RuntimeError(f"Unexpected reload status code: {status_code}")


def validate_inference_health() -> None:
    status_code = _request(f"{INFERENCE_BASE_URL}/health")
    if status_code != 200:
        raise RuntimeError(f"Unexpected health status code: {status_code}")


with DAG(
    dag_id="churn_retraining_pipeline",
    description="Retrain churn model, reload inference service cache and validate health.",
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    schedule=os.getenv("AIRFLOW_RETRAIN_SCHEDULE", "0 2 * * *"),
    catchup=False,
    tags=["mlops", "training", "churn"],
) as dag:
    wait_for_inference_task = PythonOperator(
        task_id="wait_for_inference",
        python_callable=wait_for_inference,
    )

    train_model_task = DockerOperator(
        task_id="train_model",
        image=AIRFLOW_TRAIN_IMAGE,
        command=_train_command(),
        docker_url=AIRFLOW_DOCKER_HOST,
        network_mode=AIRFLOW_DOCKER_NETWORK,
        environment=_train_container_environment(),
        working_dir="/app",
        auto_remove=True,
        mount_tmp_dir=False,
        force_pull=False,
    )

    reload_model_task = PythonOperator(
        task_id="reload_inference_model",
        python_callable=reload_model,
    )

    validate_inference_task = PythonOperator(
        task_id="validate_inference_health",
        python_callable=validate_inference_health,
    )

    (
        wait_for_inference_task
        >> train_model_task
        >> reload_model_task
        >> validate_inference_task
    )
