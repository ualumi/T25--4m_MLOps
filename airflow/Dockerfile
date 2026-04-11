FROM apache/airflow:2.10.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Layout expected by DAGs: PROJECT_ROOT=/opt/project, dags in /opt/airflow/dags
COPY config /opt/project/config
COPY services/inference-service /opt/project/services/inference-service
COPY data /opt/project/data
COPY airflow/dags /opt/airflow/dags

USER airflow
ENV PYTHONPATH="/opt/project:/opt/project/services/inference-service"
RUN pip install --no-cache-dir \
    apache-airflow-providers-docker \
    boto3 \
    pandas \
    scikit-learn \
    lightgbm \
    joblib
