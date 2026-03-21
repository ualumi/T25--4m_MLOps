from fastapi.testclient import TestClient

from src.api import app


class StubConnect:
    def execute(self, user_id):
        return type("Session", (), {"user_id": user_id, "token": "token-123"})()


class StubPredict:
    def execute(self, user_id, token, features):
        return {"score": 0.61, "label": "churn"}


def test_connect_and_predict() -> None:
    app.state.connect_use_case = StubConnect()
    app.state.predict_use_case = StubPredict()
    client = TestClient(app)

    connect_response = client.post("/connect", json={"user_id": "u-1"})
    assert connect_response.status_code == 200
    assert connect_response.json()["session_token"] == "token-123"

    predict_response = client.post(
        "/predict",
        json={
            "user_id": "u-1",
            "session_token": "token-123",
            "features": [1.0, 0.5],
        },
    )
    assert predict_response.status_code == 200
    assert predict_response.json()["label"] == "churn"

    del app.state.connect_use_case
    del app.state.predict_use_case
