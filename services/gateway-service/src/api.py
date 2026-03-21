"""HTTP API сервиса gateway."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.config import get_gateway_config  # noqa: E402
from src.application.use_cases.connect_user import ConnectUserUseCase  # noqa: E402
from src.application.use_cases.request_prediction import (  # noqa: E402
    RequestPredictionUseCase,
)
from src.infrastructure.http.inference_http_client import InferenceHttpClient  # noqa: E402
from src.infrastructure.memory.session_store import InMemorySessionStore  # noqa: E402
from src.interfaces.api.schemas import (  # noqa: E402
    ConnectRequest,
    ConnectResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(title="Gateway Service", version="1.0.0")
session_store = InMemorySessionStore()


@lru_cache(maxsize=1)
def get_connect_use_case() -> ConnectUserUseCase:
    return ConnectUserUseCase(store=session_store)


@lru_cache(maxsize=1)
def get_predict_use_case() -> RequestPredictionUseCase:
    config = get_gateway_config()
    client = InferenceHttpClient(config.inference_service_url)
    return RequestPredictionUseCase(
        store=session_store, inference=client, api_key=config.service_api_key
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

    return PredictResponse(score=float(result["score"]), label=str(result["label"]))
