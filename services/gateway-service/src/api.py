"""HTTP API сервиса gateway."""

from __future__ import annotations

import csv
import io
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import cast

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = next(
    (parent for parent in CURRENT_FILE.parents if (parent / "config").exists()),
    CURRENT_FILE.parents[1],
)
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.config import get_gateway_config  # noqa: E402
from src.application.use_cases.connect_user import ConnectUserUseCase  # noqa: E402
from src.application.use_cases.request_prediction import (  # noqa: E402
    RequestPredictionUseCase,
)
from src.infrastructure.http.inference_http_client import (  # noqa: E402
    InferenceHttpClient,
)
from src.infrastructure.postgres.session_store import PostgresSessionStore  # noqa: E402
from src.infrastructure.storage.s3_dataset_store import S3DatasetStore  # noqa: E402
from src.interfaces.api.schemas import (  # noqa: E402
    BatchPredictClient,
    BatchPredictItemResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    ConnectRequest,
    ConnectResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(title="Gateway Service", version="1.0.0")


@lru_cache(maxsize=1)
def get_session_store() -> PostgresSessionStore:
    config = get_gateway_config()
    return PostgresSessionStore(config.database_url)


@lru_cache(maxsize=1)
def get_connect_use_case() -> ConnectUserUseCase:
    return ConnectUserUseCase(store=get_session_store())


@lru_cache(maxsize=1)
def get_predict_use_case() -> RequestPredictionUseCase:
    config = get_gateway_config()
    client = InferenceHttpClient(config.inference_service_url)
    return RequestPredictionUseCase(
        store=get_session_store(), inference=client, api_key=config.service_api_key
    )


@lru_cache(maxsize=1)
def get_dataset_store() -> S3DatasetStore:
    config = get_gateway_config()
    return S3DatasetStore(
        bucket=config.prediction_results_bucket,
        prefix=config.prediction_results_prefix,
        endpoint_url=config.s3_endpoint_url,
        access_key_id=config.s3_access_key_id,
        secret_access_key=config.s3_secret_access_key,
        region=config.s3_region,
    )


@lru_cache(maxsize=1)
def get_prediction_result_store() -> S3DatasetStore:
    config = get_gateway_config()
    return S3DatasetStore(
        bucket=config.prediction_results_bucket,
        prefix=config.prediction_results_prefix,
        endpoint_url=config.s3_endpoint_url,
        access_key_id=config.s3_access_key_id,
        secret_access_key=config.s3_secret_access_key,
        region=config.s3_region,
    )


def _save_prediction_result(
    user_id: str,
    request_payload: dict[str, object],
    response_payload: dict[str, object],
    result_kind: str,
) -> str:
    result_store = getattr(app.state, "prediction_result_store", None) or (
        get_prediction_result_store()
    )
    return result_store.save_json_artifact(
        user_id=user_id,
        name="prediction-result.json",
        payload={
            "request": request_payload,
            "response": response_payload,
        },
        subfolder=result_kind,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/connect", response_model=ConnectResponse)
def connect(payload: ConnectRequest) -> ConnectResponse:
    use_case = getattr(app.state, "connect_use_case", None) or get_connect_use_case()
    session = use_case.execute(payload.user_id)
    return ConnectResponse(user_id=session.user_id, session_token=session.token)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    use_case = getattr(app.state, "predict_use_case", None) or get_predict_use_case()
    try:
        result = use_case.execute(
            user_id=payload.user_id,
            token=payload.session_token,
            features=payload.features,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response_payload = {"score": float(result["score"])}
    result_uri = _save_prediction_result(
        user_id=payload.user_id,
        request_payload=payload.model_dump(),
        response_payload=response_payload,
        result_kind="one-predict",
    )
    return PredictResponse(score=response_payload["score"], result_uri=result_uri)


def _run_batch_prediction(
    payload: BatchPredictRequest,
    dataset_uri: str | None = None,
    result_kind: str = "batch",
) -> BatchPredictResponse:
    use_case = getattr(app.state, "predict_use_case", None) or get_predict_use_case()
    try:
        result = use_case.execute_batch(
            user_id=payload.user_id,
            token=payload.session_token,
            clients=[
                {"client_id": client.client_id, "features": client.features}
                for client in payload.clients
            ],
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    predictions = [
        BatchPredictItemResponse(
            client_id=item["client_id"],
            score=item["score"],
        )
        for item in result["predictions"]
    ]
    response_payload = {
        "predictions": [prediction.model_dump() for prediction in predictions],
        "dataset_uri": dataset_uri,
    }
    result_uri = _save_prediction_result(
        user_id=payload.user_id,
        request_payload=payload.model_dump(),
        response_payload=response_payload,
        result_kind=result_kind,
    )
    return BatchPredictResponse(
        predictions=predictions,
        dataset_uri=dataset_uri,
        result_uri=result_uri,
    )


def _parse_json_clients(raw_content: bytes) -> list[BatchPredictClient]:
    try:
        payload = json.loads(raw_content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON file.") from exc

    raw_clients: list[object]
    if isinstance(payload, list):
        raw_clients = cast(list[object], payload)
    elif isinstance(payload, dict) and isinstance(payload.get("clients"), list):
        raw_clients = cast(list[object], payload["clients"])
    else:
        raise ValueError("JSON file must be an array or an object with 'clients'.")

    try:
        return [BatchPredictClient.model_validate(client) for client in raw_clients]
    except ValidationError as exc:
        raise ValueError("JSON file содержит некорректные данные клиентов.") from exc


def _parse_csv_clients(raw_content: bytes) -> list[BatchPredictClient]:
    try:
        text = raw_content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV file must be UTF-8 encoded.") from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV file must contain a header row.")
    if "client_id" not in reader.fieldnames:
        raise ValueError("CSV file must contain a 'client_id' column.")

    feature_columns = [column for column in reader.fieldnames if column != "client_id"]
    clients: list[BatchPredictClient] = []

    for row_number, row in enumerate(reader, start=2):
        client_id = (row.get("client_id") or "").strip()
        if not client_id:
            raise ValueError(f"Row {row_number} has an empty client_id.")

        try:
            features = [float(row[column]) for column in feature_columns]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Row {row_number} contains non-numeric feature values."
            ) from exc

        clients.append(BatchPredictClient(client_id=client_id, features=features))

    if not clients:
        raise ValueError("CSV file must contain at least one client row.")

    return clients


def _build_batch_request_from_upload(
    user_id: str,
    session_token: str,
    filename: str,
    raw_content: bytes,
) -> BatchPredictRequest:
    suffix = Path(filename).suffix.lower()

    if suffix == ".json":
        clients = _parse_json_clients(raw_content)
    elif suffix == ".csv":
        clients = _parse_csv_clients(raw_content)
    else:
        raise HTTPException(
            status_code=400, detail="Supported file formats are .json and .csv."
        )

    try:
        return BatchPredictRequest(
            user_id=user_id,
            session_token=session_token,
            clients=clients,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    return _run_batch_prediction(payload, result_kind="batch")


@app.post("/predict/upload", response_model=BatchPredictResponse)
async def predict_upload(
    user_id: str = Form(...),
    session_token: str = Form(...),
    file: UploadFile = File(...),
) -> BatchPredictResponse:
    raw_content = await file.read()
    filename = file.filename or "dataset"
    payload = _build_batch_request_from_upload(
        user_id=user_id,
        session_token=session_token,
        filename=filename,
        raw_content=raw_content,
    )
    dataset_store = getattr(app.state, "dataset_store", None) or get_dataset_store()
    dataset_uri = dataset_store.save_dataset(
        user_id=user_id,
        filename=filename,
        data=raw_content,
        content_type=file.content_type or "application/octet-stream",
        subfolder="uploads",
    )
    return _run_batch_prediction(
        payload,
        dataset_uri=dataset_uri,
        result_kind="uploads",
    )
