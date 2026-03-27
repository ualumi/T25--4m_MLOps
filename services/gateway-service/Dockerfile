FROM python:3.11-slim

WORKDIR /app

COPY services/gateway-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config /app/config
COPY services/gateway-service/src /app/src

ENV INFERENCE_SERVICE_URL=http://inference-service:8001
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
