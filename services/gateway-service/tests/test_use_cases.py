from src.application.use_cases.connect_user import ConnectUserUseCase
from src.application.use_cases.request_prediction import RequestPredictionUseCase
from src.infrastructure.memory.session_store import InMemorySessionStore


class StubInference:
    def predict(self, features, api_key):
        return {"score": 0.35, "label": "stay"}


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
    assert response["label"] == "stay"


def test_request_prediction_use_case_invalid_session() -> None:
    store = InMemorySessionStore()
    use_case = RequestPredictionUseCase(store, StubInference(), "k")
    try:
        use_case.execute("u-1", "bad-token", [0.1])
        assert False, "PermissionError was expected"
    except PermissionError:
        assert True
