from config.config import get_inference_config


def test_inference_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_PORT", "9001")
    monkeypatch.setenv("CHURN_THRESHOLD", "0.7")
    monkeypatch.setenv("MODEL_PATH", "abc.joblib")

    config = get_inference_config()

    assert config.port == 9001
    assert config.threshold == 0.7
    assert config.model_path == "abc.joblib"
