FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /uvx /bin/

ENV UV_SYSTEM_PYTHON=1
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/inference-service/requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY config /app/config
COPY data /app/data
COPY services/inference-service/src /app/src

ENV MODEL_URI=s3://ml-artifacts/models/production/model.joblib
EXPOSE 8001

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8001"]
