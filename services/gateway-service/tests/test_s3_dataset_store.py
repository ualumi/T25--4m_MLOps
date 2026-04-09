from src.infrastructure.storage.s3_dataset_store import S3DatasetStore


class RecordingPutClient:
    def __init__(self) -> None:
        self.last_put_kwargs: dict[str, object] | None = None

    def put_object(self, **kwargs) -> None:
        self.last_put_kwargs = kwargs


def test_save_bytes_applies_sse_aes256() -> None:
    store = S3DatasetStore(
        bucket="ml-artifacts",
        prefix="training/uploads",
        sse_mode="AES256",
    )
    fake_client = RecordingPutClient()
    store._client = fake_client  # pylint: disable=protected-access
    store.ensure_bucket = lambda: None

    uri = store.save_bytes(
        user_id="u-1",
        filename="train.csv",
        data=b"payload",
        content_type="text/csv",
        subfolder="training",
    )

    assert uri.startswith("s3://ml-artifacts/training/uploads/u-1/training/")
    assert fake_client.last_put_kwargs is not None
    assert fake_client.last_put_kwargs["ServerSideEncryption"] == "AES256"
    assert "SSEKMSKeyId" not in fake_client.last_put_kwargs


def test_save_bytes_applies_sse_kms_key() -> None:
    store = S3DatasetStore(
        bucket="ml-artifacts",
        prefix="training/uploads",
        sse_mode="aws:kms",
        sse_kms_key_id="datasets-key",
    )
    fake_client = RecordingPutClient()
    store._client = fake_client  # pylint: disable=protected-access
    store.ensure_bucket = lambda: None

    store.save_bytes(
        user_id="u-1",
        filename="train.csv",
        data=b"payload",
        content_type="text/csv",
        subfolder="training",
    )

    assert fake_client.last_put_kwargs is not None
    assert fake_client.last_put_kwargs["ServerSideEncryption"] == "aws:kms"
    assert fake_client.last_put_kwargs["SSEKMSKeyId"] == "datasets-key"
