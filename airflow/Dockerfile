FROM apache/airflow:2.10.5-python3.11

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY services/inference-service/requirements.txt /tmp/inference-requirements.txt

RUN pip install --no-cache-dir psycopg2-binary==2.9.10 \
    && pip install --no-cache-dir -r /tmp/inference-requirements.txt
