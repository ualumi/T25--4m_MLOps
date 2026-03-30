FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /uvx /bin/

ENV UV_SYSTEM_PYTHON=1
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY services/gateway-service/requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY config /app/config
COPY services/gateway-service/src /app/src

ENV INFERENCE_SERVICE_URL=http://inference-service:8001
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
