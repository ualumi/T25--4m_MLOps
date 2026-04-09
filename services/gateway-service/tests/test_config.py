from config.config import get_gateway_config


def test_gateway_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_PORT", "9010")
    monkeypatch.setenv("INFERENCE_SERVICE_URL", "http://localhost:8100")
    monkeypatch.setenv("SERVICE_API_KEY", "abc")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("S3_SSE_MODE", "aws:kms")
    monkeypatch.setenv("S3_SSE_KMS_KEY_ID", "datasets-key")
    monkeypatch.setenv("REQUIRE_UPLOAD_ENCRYPTION", "true")
    monkeypatch.setenv("DATASET_UPLOADS_BUCKET", "user-datasets")
    monkeypatch.setenv("DATASET_UPLOADS_PREFIX", "incoming")
    monkeypatch.setenv("PREDICTION_RESULTS_BUCKET", "prediction-results")
    monkeypatch.setenv("PREDICTION_RESULTS_PREFIX", "responses")
    monkeypatch.setenv("SEGMENT_RESULTS_BUCKET", "segment-results")
    monkeypatch.setenv("SEGMENT_RESULTS_PREFIX", "segments-v2")
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
    assert config.s3_sse_mode == "aws:kms"
    assert config.s3_sse_kms_key_id == "datasets-key"
    assert config.require_upload_encryption is True
    assert config.dataset_upload_bucket == "user-datasets"
    assert config.dataset_upload_prefix == "incoming"
    assert config.prediction_results_bucket == "prediction-results"
    assert config.prediction_results_prefix == "responses"
    assert config.segment_results_bucket == "segment-results"
    assert config.segment_results_prefix == "segments-v2"
