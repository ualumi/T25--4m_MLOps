from config.config import get_inference_config


def test_inference_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_PORT", "9001")
    monkeypatch.setenv("MODEL_URI", "s3://bucket/abc.joblib")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

    config = get_inference_config()

    assert config.port == 9001
    assert config.model_uri == "s3://bucket/abc.joblib"
    assert config.s3_endpoint_url == "http://minio:9000"
    assert config.s3_access_key_id == "key"
    assert config.s3_secret_access_key == "secret"
    assert config.s3_region == "eu-central-1"
