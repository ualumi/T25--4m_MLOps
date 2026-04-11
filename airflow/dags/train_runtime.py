"""Оператор обучения в DAG: DockerOperator (docker-compose)."""

from __future__ import annotations

import os

from airflow.models import BaseOperator
from airflow.providers.docker.operators.docker import DockerOperator


def make_train_operator(
    *,
    task_id: str,
    name: str,
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
    )
