from fastapi.testclient import TestClient

from src.api import app


class StubUseCase:
    def execute(self, payload):
        return type("Result", (), {"score": 0.88, "label": "churn"})()


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
            )()
        ]
        return type(
            "SegmentResult",
            (),
            {"top_share": top_share, "total_users": len(candidates), "segment": ranked},
        )()


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint() -> None:
    app.state.use_case = StubUseCase()
    client = TestClient(app)

    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})

    assert response.status_code == 200
    assert response.json()["label"] == "churn"
    del app.state.use_case


def test_segment_endpoint() -> None:
    app.state.segment_use_case = StubSegmentUseCase()
    client = TestClient(app)

    response = client.post(
        "/segment",
        json={
            "users": [
                {"user_id": "u-1", "features": [0.1, 0.2]},
                {"user_id": "u-2", "features": [1.0, 0.8]},
            ],
            "top_share": 0.2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["segment_size"] == 1
    assert body["segment"][0]["user_id"] == "u-2"
    assert body["segment"][0]["rank"] == 1
    del app.state.segment_use_case
