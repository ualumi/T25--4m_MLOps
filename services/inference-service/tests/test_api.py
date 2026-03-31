from fastapi.testclient import TestClient

from src.api import app

VALID_FEATURES = [0.1] * 28


class StubUseCase:
    def execute(self, payload):
        return type("Result", (), {"score": 0.88})()

    def execute_batch(self, clients):
        return type(
            "BatchResult",
            (),
            {
                "predictions": [
                    type(
                        "Prediction",
                        (),
                        {"client_id": client.client_id, "score": 0.88},
                    )()
                    for client in clients
                ]
            },
        )()


class StubSegmentUseCase:
    def execute(self, candidates, top_share):
        ranked = [
            type(
                "Ranked",
                (),
                {
                    "user_id": "u-2",
                    "probability_of_inactivity": 0.9,
                    "rank": 1,
                },
            )(),
            type(
                "Ranked",
                (),
                {
                    "user_id": "u-1",
                    "probability_of_inactivity": 0.2,
                    "rank": 2,
                },
            )(),
        ]
        return type(
            "SegmentResult",
            (),
            {
                "top_share": top_share,
                "total_users": len(candidates),
                "segment": ranked,
                "top_segment": ranked[:1],
            },
        )()


class StubArtifactStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []

    def save_json(self, uri, payload):
        self.saved.append({"uri": uri, "payload": payload})


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint() -> None:
    app.state.use_case = StubUseCase()
    client = TestClient(app)

    response = client.post("/predict", json={"features": VALID_FEATURES})

    assert response.status_code == 200
    assert response.json()["score"] == 0.88
    del app.state.use_case


def test_predict_batch_endpoint() -> None:
    app.state.use_case = StubUseCase()
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        json={
            "clients": [
                {"client_id": "c-1", "features": VALID_FEATURES},
                {"client_id": "c-2", "features": VALID_FEATURES},
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["client_id"] == "c-1"
    assert body["predictions"][0]["score"] == 0.88
    del app.state.use_case


def test_predict_batch_endpoint_rejects_invalid_feature_count() -> None:
    app.state.use_case = StubUseCase()
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        json={
            "clients": [
                {"client_id": "c-1", "features": [0.1, 0.2, 0.3]},
            ]
        },
    )

    assert response.status_code == 422
    del app.state.use_case


def test_segment_endpoint() -> None:
    app.state.segment_use_case = StubSegmentUseCase()
    app.state.artifact_store = StubArtifactStore()
    client = TestClient(app)

    response = client.post(
        "/segment",
        json={
            "users": [
                {"user_id": "u-1", "features": VALID_FEATURES},
                {"user_id": "u-2", "features": VALID_FEATURES},
            ],
            "top_share": 0.2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["top_share"] == 0.2
    assert body["total_users"] == 2
    assert len(body["segment"]) == 2
    assert body["top_segment_size"] == 1
    assert body["segment"][0]["user_id"] == "u-2"
    assert body["segment"][0]["rank"] == 1
    assert body["top_segment"][0]["user_id"] == "u-2"
    assert body["result_uri"].startswith("s3://")
    assert len(app.state.artifact_store.saved) == 1
    metadata = app.state.artifact_store.saved[0]["payload"]["metadata"]
    assert metadata["endpoint"] == "/segment"
    assert metadata["record_count"] == 2
    del app.state.segment_use_case
    del app.state.artifact_store
