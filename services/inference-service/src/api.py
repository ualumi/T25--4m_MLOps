"""HTTP API сервиса инференса."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = next(
    (parent for parent in CURRENT_FILE.parents if (parent / "config").exists()),
    CURRENT_FILE.parents[1],
)
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.config import get_inference_config  # noqa: E402
from src.application.use_cases.build_target_segment import (  # noqa: E402
    BuildTargetSegmentUseCase,
)
from src.application.use_cases.predict_churn import (  # noqa: E402
    BatchPredictClientPayload,
    PredictChurnUseCase,
)
from src.domain.entities.prediction_input import PredictionInput  # noqa: E402
from src.domain.entities.scoring_candidate import ScoringCandidate  # noqa: E402
from src.infrastructure.ml.joblib_model_scorer import JoblibModelScorer  # noqa: E402
from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore  # noqa: E402
from src.interfaces.api.schemas import (  # noqa: E402
    BatchPredictItemResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    SegmentRequest,
    SegmentResponse,
    SegmentUserResponse,
)

app = FastAPI(title="Inference Service", version="1.0.0")


@lru_cache(maxsize=1)
def build_artifact_store() -> S3ArtifactStore:
    config = get_inference_config()
    return S3ArtifactStore(
        endpoint_url=config.s3_endpoint_url,
        access_key_id=config.s3_access_key_id,
        secret_access_key=config.s3_secret_access_key,
        region=config.s3_region,
    )


@lru_cache(maxsize=1)
def build_use_case() -> PredictChurnUseCase:
    config = get_inference_config()
    scorer = JoblibModelScorer(config.model_uri, build_artifact_store())
    return PredictChurnUseCase(scorer=scorer)


@lru_cache(maxsize=1)
def build_segment_use_case() -> BuildTargetSegmentUseCase:
    config = get_inference_config()
    scorer = JoblibModelScorer(config.model_uri, build_artifact_store())
    return BuildTargetSegmentUseCase(scorer=scorer)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    use_case = getattr(app.state, "use_case", None) or build_use_case()
    try:
        result = use_case.execute(PredictionInput(features=payload.features))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(score=result.score)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    use_case = getattr(app.state, "use_case", None) or build_use_case()
    try:
        result = use_case.execute_batch(
            clients=[
                BatchPredictClientPayload(
                    client_id=client.client_id,
                    features=client.features,
                )
                for client in payload.clients
            ]
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return BatchPredictResponse(
        predictions=[
            BatchPredictItemResponse(client_id=item.client_id, score=item.score)
            for item in result.predictions
        ]
    )


@app.post("/segment", response_model=SegmentResponse)
def build_segment(payload: SegmentRequest) -> SegmentResponse:
    use_case = getattr(app.state, "segment_use_case", None) or build_segment_use_case()
    try:
        result = use_case.execute(
            candidates=[
                ScoringCandidate(user_id=user.user_id, features=user.features)
                for user in payload.users
            ],
            top_share=payload.top_share,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SegmentResponse(
        top_share=result.top_share,
        total_users=result.total_users,
        segment=[
            SegmentUserResponse(
                user_id=user.user_id,
                probability_of_inactivity=user.probability_of_inactivity,
                rank=user.rank,
            )
            for user in result.segment
        ],
        top_segment_size=len(result.top_segment),
        top_segment=[
            SegmentUserResponse(
                user_id=user.user_id,
                probability_of_inactivity=user.probability_of_inactivity,
                rank=user.rank,
            )
            for user in result.top_segment
        ],
    )
