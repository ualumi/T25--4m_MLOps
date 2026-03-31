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

    def execute_segment(self, user_id, token, users, top_share):
        return {
            "top_share": top_share,
            "total_users": len(users),
            "segment": [
                {
                    "user_id": "c-2",
                    "probability_of_inactivity": 0.91,
                    "rank": 1,
                },
                {
                    "user_id": "c-1",
                    "probability_of_inactivity": 0.45,
                    "rank": 2,
                },
            ],
            "top_segment_size": 1,
            "top_segment": [
                {
                    "user_id": "c-2",
                    "probability_of_inactivity": 0.91,
                    "rank": 1,
                }
            ],
            "result_uri": "s3://ml-artifacts/segments/result.json",
        }


class StubDatasetStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, str | bytes]] = []

    def save_dataset(self, user_id, filename, data, content_type, subfolder=None):
        self.saved.append(
            {
                "user_id": user_id,
                "filename": filename,
                "data": data,
                "content_type": content_type,
                "subfolder": subfolder,
            }
        )
        return f"s3://ml-artifacts/prediction-results/{user_id}/{subfolder}/{filename}"


class StubPredictionResultStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []

    def save_json_artifact(self, user_id, name, payload, subfolder=None):
        self.saved.append(
            {
                "user_id": user_id,
                "name": name,
                "payload": payload,
                "subfolder": subfolder,
            }
        )
        return f"s3://ml-artifacts/prediction-results/{user_id}/{subfolder}/{name}"


def test_connect_and_predict() -> None:
    app.state.connect_use_case = StubConnect()
    app.state.predict_use_case = StubPredict()
    app.state.prediction_result_store = StubPredictionResultStore()
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
    assert (
        predict_response.json()["result_uri"]
        == "s3://ml-artifacts/prediction-results/u-1/one-predict/prediction-result.json"
    )
    assert app.state.prediction_result_store.saved[0]["subfolder"] == "one-predict"
    metadata = app.state.prediction_result_store.saved[0]["payload"]["metadata"]
    assert metadata["endpoint"] == "/predict"
    assert metadata["record_count"] == 1
    assert metadata["created_at"]

    del app.state.connect_use_case
    del app.state.predict_use_case
    del app.state.prediction_result_store


def test_predict_batch() -> None:
    app.state.predict_use_case = StubPredict()
    app.state.prediction_result_store = StubPredictionResultStore()
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
    assert (
        body["result_uri"]
        == "s3://ml-artifacts/prediction-results/u-1/batch/prediction-result.json"
    )
    assert app.state.prediction_result_store.saved[0]["subfolder"] == "batch"
    metadata = app.state.prediction_result_store.saved[0]["payload"]["metadata"]
    assert metadata["endpoint"] == "/predict/batch"
    assert metadata["record_count"] == 2

    del app.state.predict_use_case
    del app.state.prediction_result_store


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
    app.state.dataset_store = StubDatasetStore()
    app.state.prediction_result_store = StubPredictionResultStore()
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
    assert (
        body["dataset_uri"]
        == "s3://ml-artifacts/prediction-results/u-1/uploads/clients.json"
    )
    assert (
        body["result_uri"]
        == "s3://ml-artifacts/prediction-results/u-1/uploads/prediction-result.json"
    )
    assert app.state.dataset_store.saved[0]["filename"] == "clients.json"
    assert app.state.dataset_store.saved[0]["subfolder"] == "uploads"
    assert app.state.prediction_result_store.saved[0]["subfolder"] == "uploads"
    metadata = app.state.prediction_result_store.saved[0]["payload"]["metadata"]
    assert metadata["endpoint"] == "/predict/upload"
    assert metadata["record_count"] == 2

    del app.state.predict_use_case
    del app.state.dataset_store
    del app.state.prediction_result_store


def test_predict_upload_csv() -> None:
    app.state.predict_use_case = StubPredict()
    app.state.dataset_store = StubDatasetStore()
    app.state.prediction_result_store = StubPredictionResultStore()
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
    assert (
        body["dataset_uri"]
        == "s3://ml-artifacts/prediction-results/u-1/uploads/clients.csv"
    )
    assert (
        body["result_uri"]
        == "s3://ml-artifacts/prediction-results/u-1/uploads/prediction-result.json"
    )
    assert app.state.dataset_store.saved[0]["content_type"] == "text/csv"
    assert app.state.prediction_result_store.saved[0]["subfolder"] == "uploads"
    metadata = app.state.prediction_result_store.saved[0]["payload"]["metadata"]
    assert metadata["endpoint"] == "/predict/upload"
    assert metadata["record_count"] == 2

    del app.state.predict_use_case
    del app.state.dataset_store
    del app.state.prediction_result_store


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


def test_segment() -> None:
    app.state.predict_use_case = StubPredict()
    app.state.prediction_result_store = StubPredictionResultStore()
    client = TestClient(app)

    response = client.post(
        "/segment",
        json={
            "user_id": "u-1",
            "session_token": "token-123",
            "users": [
                {"user_id": "c-1", "features": VALID_FEATURES},
                {"user_id": "c-2", "features": VALID_FEATURES},
            ],
            "top_share": 0.5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["top_share"] == 0.5
    assert body["top_segment_size"] == 1
    assert body["segment"][0]["user_id"] == "c-2"
    assert (
        body["result_uri"]
        == "s3://ml-artifacts/prediction-results/u-1/segments/prediction-result.json"
    )
    assert app.state.prediction_result_store.saved[0]["subfolder"] == "segments"
    metadata = app.state.prediction_result_store.saved[0]["payload"]["metadata"]
    assert metadata["endpoint"] == "/segment"
    assert metadata["record_count"] == 2

    del app.state.predict_use_case
    del app.state.prediction_result_store
