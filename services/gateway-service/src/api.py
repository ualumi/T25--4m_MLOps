"""HTTP API сервиса gateway."""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Mapping, cast

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
    SegmentRequest,
    SegmentResponse,
    SegmentUserResponse,
)

app = FastAPI(title="Gateway Service", version="1.0.0")
_logger = logging.getLogger(__name__)


def _validate_upload_encryption_config() -> None:
    config = get_gateway_config()
    if not config.require_upload_encryption:
        return

    mode = (config.s3_sse_mode or "").strip()
    if mode not in {"AES256", "aws:kms"}:
        raise RuntimeError(
            "Upload encryption is required. Set S3_SSE_MODE to AES256 or aws:kms."
        )
    if mode == "aws:kms" and not (config.s3_sse_kms_key_id or "").strip():
        raise RuntimeError(
            "Upload encryption is required. Set S3_SSE_KMS_KEY_ID for aws:kms mode."
        )


@app.on_event("startup")
def _on_gateway_startup() -> None:
    _validate_upload_encryption_config()
    cfg = get_gateway_config()
    if cfg.s3_endpoint_url:
        _logger.info(
            "S3: endpoint=%s — training uploads → s3://%s/%s/<uuid>-<file>",
            cfg.s3_endpoint_url,
            cfg.retrain_source_bucket,
            cfg.retrain_source_prefix,
        )
    else:
        _logger.warning(
            "S3_ENDPOINT_URL не задан: boto3 будет обращаться к публичному AWS S3, "
            "а не к MinIO. Для Docker: S3_ENDPOINT_URL=http://minio:9000; "
            "с хоста: http://127.0.0.1:9000"
        )


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
    _validate_upload_encryption_config()
    config = get_gateway_config()
    return S3DatasetStore(
        bucket=config.dataset_upload_bucket,
        prefix=config.dataset_upload_prefix,
        endpoint_url=config.s3_endpoint_url,
        access_key_id=config.s3_access_key_id,
        secret_access_key=config.s3_secret_access_key,
        region=config.s3_region,
        sse_mode=config.s3_sse_mode,
        sse_kms_key_id=config.s3_sse_kms_key_id,
    )


@lru_cache(maxsize=1)
def get_retraining_dataset_store() -> S3DatasetStore:
    _validate_upload_encryption_config()
    config = get_gateway_config()
    return S3DatasetStore(
        bucket=config.retrain_source_bucket,
        prefix=config.retrain_source_prefix,
        endpoint_url=config.s3_endpoint_url,
        access_key_id=config.s3_access_key_id,
        secret_access_key=config.s3_secret_access_key,
        region=config.s3_region,
        sse_mode=config.s3_sse_mode,
        sse_kms_key_id=config.s3_sse_kms_key_id,
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
        sse_mode=config.s3_sse_mode,
        sse_kms_key_id=config.s3_sse_kms_key_id,
    )


@lru_cache(maxsize=1)
def get_segment_result_store() -> S3DatasetStore:
    config = get_gateway_config()
    return S3DatasetStore(
        bucket=config.segment_results_bucket,
        prefix=config.segment_results_prefix,
        endpoint_url=config.s3_endpoint_url,
        access_key_id=config.s3_access_key_id,
        secret_access_key=config.s3_secret_access_key,
        region=config.s3_region,
        sse_mode=config.s3_sse_mode,
        sse_kms_key_id=config.s3_sse_kms_key_id,
    )


def _save_prediction_result(
    user_id: str,
    request_payload: Mapping[str, object],
    response_payload: Mapping[str, object],
    result_kind: str,
    endpoint: str,
    record_count: int,
    dataset_encryption: Mapping[str, object] | None = None,
) -> str:
    result_store = getattr(app.state, "prediction_result_store", None) or (
        get_prediction_result_store()
    )
    return result_store.save_json_artifact(
        user_id=user_id,
        name="prediction-result.json",
        payload={
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "record_count": record_count,
                "dataset_encryption": dataset_encryption or {},
            },
            "request": request_payload,
            "response": response_payload,
        },
        subfolder=result_kind,
    )


def _save_segment_result(
    user_id: str,
    request_payload: Mapping[str, object],
    response_payload: Mapping[str, object],
    record_count: int,
) -> str:
    result_store = getattr(app.state, "segment_result_store", None) or (
        get_segment_result_store()
    )
    return result_store.save_json_artifact(
        user_id=user_id,
        name="segment-result.json",
        payload={
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": "/segment",
                "record_count": record_count,
            },
            "request": request_payload,
            "response": response_payload,
        },
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
        endpoint="/predict",
        record_count=1,
    )
    return PredictResponse(score=response_payload["score"], result_uri=result_uri)


def _run_batch_prediction(
    payload: BatchPredictRequest,
    dataset_uri: str | None = None,
    result_kind: str = "batch",
    dataset_encryption: Mapping[str, object] | None = None,
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
        endpoint="/predict/upload" if result_kind == "by-upload" else "/predict/batch",
        record_count=len(payload.clients),
        dataset_encryption=dataset_encryption,
    )
    return BatchPredictResponse(
        predictions=predictions,
        dataset_uri=dataset_uri,
        result_uri=result_uri,
    )


@app.post("/segment", response_model=SegmentResponse)
def segment(payload: SegmentRequest) -> SegmentResponse:
    use_case = getattr(app.state, "predict_use_case", None) or get_predict_use_case()
    try:
        result = use_case.execute_segment(
            user_id=payload.user_id,
            token=payload.session_token,
            users=[
                {"user_id": user.user_id, "features": user.features}
                for user in payload.users
            ],
            top_share=payload.top_share,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    segment_users = [
        SegmentUserResponse(
            user_id=user["user_id"],
            probability_of_inactivity=user["probability_of_inactivity"],
            rank=user["rank"],
        )
        for user in result["segment"]
    ]
    top_segment_users = [
        SegmentUserResponse(
            user_id=user["user_id"],
            probability_of_inactivity=user["probability_of_inactivity"],
            rank=user["rank"],
        )
        for user in result["top_segment"]
    ]
    response_payload = {
        "top_share": result["top_share"],
        "total_users": result["total_users"],
        "segment": [user.model_dump() for user in segment_users],
        "top_segment_size": result["top_segment_size"],
        "top_segment": [user.model_dump() for user in top_segment_users],
    }
    result_uri = _save_segment_result(
        user_id=payload.user_id,
        request_payload=payload.model_dump(),
        response_payload=response_payload,
        record_count=len(payload.users),
    )
    return SegmentResponse(
        top_share=result["top_share"],
        total_users=result["total_users"],
        segment=segment_users,
        top_segment_size=result["top_segment_size"],
        top_segment=top_segment_users,
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


def _validate_training_csv(raw_content: bytes) -> int:
    try:
        text = raw_content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="CSV file must be UTF-8 encoded."
        ) from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(
            status_code=400, detail="CSV file must contain a header row."
        )
    if "user_id" not in reader.fieldnames:
        raise HTTPException(
            status_code=400,
            detail="Training CSV file must contain a 'user_id' column.",
        )
    if "churn" not in reader.fieldnames:
        raise HTTPException(
            status_code=400,
            detail="Training CSV file must contain a 'churn' column.",
        )

    row_count = 0
    for row_number, row in enumerate(reader, start=2):
        user_id = (row.get("user_id") or "").strip()
        churn = (row.get("churn") or "").strip()
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail=f"Row {row_number} has an empty user_id.",
            )
        if churn not in {"0", "1"}:
            raise HTTPException(
                status_code=400,
                detail=f"Row {row_number} must contain churn value 0 or 1.",
            )
        row_count += 1

    if row_count < 1:
        raise HTTPException(
            status_code=400,
            detail="Training CSV file must contain at least one data row.",
        )
    return row_count


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    return _run_batch_prediction(
        payload,
        result_kind="batch",
        dataset_encryption={"enabled": False},
    )


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
    )
    return _run_batch_prediction(
        payload,
        dataset_uri=dataset_uri,
        result_kind="by-upload",
        dataset_encryption={
            "enabled": bool(get_gateway_config().s3_sse_mode),
            "mode": get_gateway_config().s3_sse_mode,
            "kms_key_id": get_gateway_config().s3_sse_kms_key_id,
        },
    )


@app.post("/datasets/upload-training")
async def upload_training_dataset(
    user_id: str = Form(...),
    session_token: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, object]:
    session_store = getattr(app.state, "session_store", None) or get_session_store()
    if not session_store.is_valid(user_id=user_id, token=session_token):
        raise HTTPException(status_code=401, detail="Invalid session. Connect first.")

    raw_content = await file.read()
    filename = file.filename or "training-dataset.csv"
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Supported training dataset format is .csv.",
        )

    record_count = _validate_training_csv(raw_content)
    dataset_store = getattr(app.state, "retraining_dataset_store", None) or (
        get_retraining_dataset_store()
    )
    dataset_uri = dataset_store.save_dataset_flat(
        filename=filename,
        data=raw_content,
        content_type=file.content_type or "text/csv",
    )
    _logger.info("Training dataset uploaded: %s", dataset_uri)
    return {
        "dataset_uri": dataset_uri,
        "record_count": record_count,
        "status": "uploaded",
        "dataset_encryption": {
            "enabled": bool(get_gateway_config().s3_sse_mode),
            "mode": get_gateway_config().s3_sse_mode,
            "kms_key_id": get_gateway_config().s3_sse_kms_key_id,
        },
    }
