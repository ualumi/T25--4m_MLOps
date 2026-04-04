import json

from src.infrastructure.ml.joblib_model_scorer import JoblibModelScorer
from src.infrastructure.storage.s3_artifact_store import S3ArtifactStore, parse_s3_uri


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


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {
            ("ml-artifacts", "datasets/a.csv"): b"a",
            ("ml-artifacts", "datasets/b.csv"): b"b",
        }

    def get_paginator(self, _name: str):
        client = self

        class _Paginator:
            def paginate(self, bucket_name: str, prefix: str):
                contents = [
                    {"Key": key}
                    for stored_bucket, key in client.objects
                    if stored_bucket == bucket_name and key.startswith(prefix)
                ]
                return [{"Contents": contents}]

        return _Paginator()

    def copy(self, copy_source, bucket: str, key: str) -> None:
        source = (copy_source["Bucket"], copy_source["Key"])
        self.objects[(bucket, key)] = self.objects[source]

    def delete_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)


def test_parse_s3_uri() -> None:
    bucket, key = parse_s3_uri("s3://ml-artifacts/models/production/model.joblib")
    assert bucket == "ml-artifacts"
    assert key == "models/production/model.joblib"


def test_joblib_model_scorer_loads_model_from_s3() -> None:
    artifact_store = FakeArtifactStore()
    scorer = JoblibModelScorer(
        "s3://ml-artifacts/models/production/model.joblib",
        artifact_store,
    )

    score = scorer.score([0.1, 0.2])

    assert score == 0.8
    assert (
        artifact_store.loaded_uri == "s3://ml-artifacts/models/production/model.joblib"
    )


def test_s3_artifact_store_lists_uris() -> None:
    artifact_store = S3ArtifactStore()
    fake_client = FakeS3Client()
    artifact_store._client = fake_client  # pylint: disable=protected-access

    uris = artifact_store.list_uris("ml-artifacts", "datasets")

    assert uris == [
        "s3://ml-artifacts/datasets/a.csv",
        "s3://ml-artifacts/datasets/b.csv",
    ]


def test_s3_artifact_store_moves_uri() -> None:
    artifact_store = S3ArtifactStore()
    fake_client = FakeS3Client()
    artifact_store._client = fake_client  # pylint: disable=protected-access
    artifact_store.ensure_bucket = lambda bucket: None

    artifact_store.move_uri(
        "s3://ml-artifacts/datasets/a.csv",
        "s3://ml-artifacts/archive/a.csv",
    )

    assert ("ml-artifacts", "datasets/a.csv") not in fake_client.objects
    assert fake_client.objects[("ml-artifacts", "archive/a.csv")] == b"a"


def test_s3_artifact_store_load_json_requires_object(monkeypatch) -> None:
    artifact_store = S3ArtifactStore()
    monkeypatch.setattr(
        artifact_store,
        "download_bytes",
        lambda _uri: json.dumps(["not-an-object"]).encode("utf-8"),
    )

    try:
        artifact_store.load_json(
            "s3://ml-artifacts/models/registry/model_registry.json"
        )
        assert False, "ValueError was expected"
    except ValueError as exc:
        assert "must contain an object" in str(exc)
