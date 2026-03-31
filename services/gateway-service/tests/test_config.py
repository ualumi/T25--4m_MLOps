from config.config import get_gateway_config


def test_gateway_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_PORT", "9010")
    monkeypatch.setenv("INFERENCE_SERVICE_URL", "http://localhost:8100")
    monkeypatch.setenv("SERVICE_API_KEY", "abc")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("DATASET_UPLOADS_BUCKET", "user-datasets")
    monkeypatch.setenv("DATASET_UPLOADS_PREFIX", "incoming")
    monkeypatch.setenv(
        "GATEWAY_DATABASE_URL",
        "postgresql://user:pass@localhost:5432/test_db",
    )

    config = get_gateway_config()

    assert config.port == 9010
    assert config.inference_service_url == "http://localhost:8100"
    assert config.service_api_key == "abc"
    assert config.database_url == "postgresql://user:pass@localhost:5432/test_db"
    assert config.s3_endpoint_url == "http://localhost:9000"
    assert config.s3_access_key_id == "minioadmin"
    assert config.s3_secret_access_key == "secret"
    assert config.s3_region == "us-east-1"
    assert config.dataset_upload_bucket == "user-datasets"
    assert config.dataset_upload_prefix == "incoming"
