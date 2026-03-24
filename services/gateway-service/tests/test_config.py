from config.config import get_gateway_config


def test_gateway_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_PORT", "9010")
    monkeypatch.setenv("INFERENCE_SERVICE_URL", "http://localhost:8100")
    monkeypatch.setenv("SERVICE_API_KEY", "abc")
    monkeypatch.setenv(
        "GATEWAY_DATABASE_URL",
        "postgresql://user:pass@localhost:5432/test_db",
    )

    config = get_gateway_config()

    assert config.port == 9010
    assert config.inference_service_url == "http://localhost:8100"
    assert config.service_api_key == "abc"
    assert config.database_url == "postgresql://user:pass@localhost:5432/test_db"
