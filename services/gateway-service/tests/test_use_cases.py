from src.application.use_cases.connect_user import ConnectUserUseCase
from src.application.use_cases.request_prediction import RequestPredictionUseCase
from src.infrastructure.memory.session_store import InMemorySessionStore


class StubInference:
    def predict(self, features, api_key):
        return {"score": 0.35}

    def predict_batch(self, clients, api_key):
        return {
            "predictions": [
                {"client_id": client["client_id"], "score": 0.35} for client in clients
            ]
        }


def test_connect_user_use_case() -> None:
    store = InMemorySessionStore()
    use_case = ConnectUserUseCase(store=store)
    session = use_case.execute("u-1")
    assert session.user_id == "u-1"
    assert len(session.token) > 10


def test_request_prediction_use_case() -> None:
    store = InMemorySessionStore()
    session = store.create("u-1")
    use_case = RequestPredictionUseCase(store, StubInference(), "k")
    response = use_case.execute("u-1", session.token, [0.1, 0.2])
    assert response["score"] == 0.35


def test_request_prediction_use_case_invalid_session() -> None:
    store = InMemorySessionStore()
    use_case = RequestPredictionUseCase(store, StubInference(), "k")
    try:
        use_case.execute("u-1", "bad-token", [0.1])
        assert False, "PermissionError was expected"
    except PermissionError:
        assert True


def test_request_prediction_use_case_batch() -> None:
    store = InMemorySessionStore()
    session = store.create("u-1")
    use_case = RequestPredictionUseCase(store, StubInference(), "k")
    response = use_case.execute_batch(
        "u-1",
        session.token,
        [
            {"client_id": "c-1", "features": [0.1, 0.2]},
            {"client_id": "c-2", "features": [0.3, 0.4]},
        ],
    )
    assert len(response["predictions"]) == 2
    assert response["predictions"][0]["client_id"] == "c-1"
