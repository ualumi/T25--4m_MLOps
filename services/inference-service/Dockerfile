FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/inference-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config /app/config
COPY data /app/data
COPY services/inference-service/src /app/src

ENV MODEL_URI=s3://ml-artifacts/models/lgb_model.joblib
EXPOSE 8001

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8001"]
