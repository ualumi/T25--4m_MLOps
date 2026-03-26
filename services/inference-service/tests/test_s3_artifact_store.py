from src.infrastructure.ml.joblib_model_scorer import JoblibModelScorer
from src.infrastructure.storage.s3_artifact_store import parse_s3_uri


class FakeModel:
    def predict_proba(self, rows):
        assert rows == [[0.1, 0.2]]
        return [[0.2, 0.8]]


class FakeArtifactStore:
    def __init__(self) -> None:
        self.loaded_uri: str | None = None

    def load_joblib(self, uri: str):
        self.loaded_uri = uri
        return FakeModel()


def test_parse_s3_uri() -> None:
    bucket, key = parse_s3_uri("s3://ml-artifacts/models/lgb_model.joblib")
    assert bucket == "ml-artifacts"
    assert key == "models/lgb_model.joblib"


def test_joblib_model_scorer_loads_model_from_s3() -> None:
    artifact_store = FakeArtifactStore()
    scorer = JoblibModelScorer(
        "s3://ml-artifacts/models/lgb_model.joblib",
        artifact_store,
    )

    score = scorer.score([0.1, 0.2])

    assert score == 0.8
    assert artifact_store.loaded_uri == "s3://ml-artifacts/models/lgb_model.joblib"
