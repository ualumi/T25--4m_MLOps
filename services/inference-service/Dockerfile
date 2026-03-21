FROM python:3.11-slim

WORKDIR /app

COPY services/inference-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common
COPY services/inference-service/src /app/src
COPY models /app/models

ENV MODEL_PATH=/app/models/lgb_model.joblib
EXPOSE 8001

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8001"]
