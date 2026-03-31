import json

from fastapi.testclient import TestClient

from src.api import app

VALID_FEATURES = [0.1] * 28


class StubConnect:
    def execute(self, user_id):
        return type("Session", (), {"user_id": user_id, "token": "token-123"})()


class StubPredict:
    def execute(self, user_id, token, features):
        return {"score": 0.61}

    def execute_batch(self, user_id, token, clients):
        return {
            "predictions": [
                {"client_id": client["client_id"], "score": 0.61} for client in clients
            ]
        }


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
            "features": VALID_FEATURES,
        },
    )
    assert predict_response.status_code == 200
    assert predict_response.json()["score"] == 0.61

    del app.state.connect_use_case
    del app.state.predict_use_case


def test_predict_batch() -> None:
    app.state.predict_use_case = StubPredict()
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        json={
            "user_id": "u-1",
            "session_token": "token-123",
            "clients": [
                {"client_id": "c-1", "features": VALID_FEATURES},
                {"client_id": "c-2", "features": VALID_FEATURES},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["client_id"] == "c-1"
    assert body["predictions"][0]["score"] == 0.61

    del app.state.predict_use_case


def test_predict_batch_rejects_invalid_feature_count() -> None:
    app.state.predict_use_case = StubPredict()
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        json={
            "user_id": "u-1",
            "session_token": "token-123",
            "clients": [
                {"client_id": "c-1", "features": [1.0, 0.5]},
            ],
        },
    )

    assert response.status_code == 422

    del app.state.predict_use_case


def test_predict_upload_json() -> None:
    app.state.predict_use_case = StubPredict()
    client = TestClient(app)

    response = client.post(
        "/predict/upload",
        data={"user_id": "u-1", "session_token": "token-123"},
        files={
            "file": (
                "clients.json",
                json.dumps(
                    {
                        "clients": [
                            {"client_id": "c-1", "features": VALID_FEATURES},
                            {"client_id": "c-2", "features": VALID_FEATURES},
                        ]
                    }
                ),
                "application/json",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["client_id"] == "c-1"

    del app.state.predict_use_case


def test_predict_upload_csv() -> None:
    app.state.predict_use_case = StubPredict()
    client = TestClient(app)
    header = ["client_id", *[f"feature_{index}" for index in range(1, 29)]]
    row_one = ["c-1", *["0.1"] * 28]
    row_two = ["c-2", *["0.2"] * 28]
    csv_content = "\n".join(
        [
            ",".join(header),
            ",".join(row_one),
            ",".join(row_two),
        ]
    )

    response = client.post(
        "/predict/upload",
        data={"user_id": "u-1", "session_token": "token-123"},
        files={"file": ("clients.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][1]["client_id"] == "c-2"

    del app.state.predict_use_case


def test_predict_upload_rejects_unsupported_file_type() -> None:
    app.state.predict_use_case = StubPredict()
    client = TestClient(app)

    response = client.post(
        "/predict/upload",
        data={"user_id": "u-1", "session_token": "token-123"},
        files={"file": ("clients.txt", "test", "text/plain")},
    )

    assert response.status_code == 400

    del app.state.predict_use_case
