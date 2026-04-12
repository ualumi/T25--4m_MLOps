from src.infrastructure.storage.s3_dataset_store import S3DatasetStore


class RecordingPutClient:
    def __init__(self) -> None:
        self.last_put_kwargs: dict[str, object] | None = None

    def put_object(self, **kwargs) -> None:
        self.last_put_kwargs = kwargs


def test_save_bytes_applies_sse_aes256() -> None:
    store = S3DatasetStore(
        bucket="ml-artifacts",
        prefix="inference/uploads",
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
    )

    assert uri.startswith("s3://ml-artifacts/inference/uploads/u-1/")
    assert fake_client.last_put_kwargs is not None
    assert fake_client.last_put_kwargs["ServerSideEncryption"] == "AES256"
    assert "SSEKMSKeyId" not in fake_client.last_put_kwargs


def test_save_bytes_applies_sse_kms_key() -> None:
    store = S3DatasetStore(
        bucket="ml-artifacts",
        prefix="inference/uploads",
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
    )

    assert fake_client.last_put_kwargs is not None
    assert fake_client.last_put_kwargs["ServerSideEncryption"] == "aws:kms"
    assert fake_client.last_put_kwargs["SSEKMSKeyId"] == "datasets-key"


def test_save_dataset_flat_stores_only_file_under_prefix() -> None:
    store = S3DatasetStore(
        bucket="ml-artifacts",
        prefix="training",
        sse_mode="AES256",
    )
    fake_client = RecordingPutClient()
    store._client = fake_client  # pylint: disable=protected-access
    store.ensure_bucket = lambda: None

    uri = store.save_dataset_flat(
        filename="train.csv",
        data=b"payload",
        content_type="text/csv",
    )

    assert uri.startswith("s3://ml-artifacts/training/")
    relative = uri.removeprefix("s3://ml-artifacts/training/")
    assert "/" not in relative
    assert relative.endswith("-train.csv")

    assert fake_client.last_put_kwargs is not None
    assert fake_client.last_put_kwargs["ServerSideEncryption"] == "AES256"
    assert "SSEKMSKeyId" not in fake_client.last_put_kwargs
