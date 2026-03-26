"""S3-совместимое хранилище артефактов модели."""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.parse import urlparse

import boto3
import joblib
from botocore.exceptions import ClientError


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


class S3ArtifactStore:
    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
        )
        self._region = region

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._client.head_bucket(Bucket=bucket)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code not in {"404", "NoSuchBucket"}:
                raise

            create_kwargs: dict[str, Any] = {"Bucket": bucket}
            if self._region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._region
                }
            self._client.create_bucket(**create_kwargs)

    def download_bytes(self, uri: str) -> bytes:
        bucket, key = parse_s3_uri(uri)
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NoSuchBucket"}:
                raise FileNotFoundError(f"S3 artifact not found: {uri}") from exc
            raise
        return response["Body"].read()

    def upload_bytes(
        self,
        uri: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        bucket, key = parse_s3_uri(uri)
        self.ensure_bucket(bucket)
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    def load_joblib(self, uri: str) -> Any:
        return joblib.load(io.BytesIO(self.download_bytes(uri)))

    def save_joblib(self, uri: str, artifact: Any) -> None:
        buffer = io.BytesIO()
        joblib.dump(artifact, buffer)
        self.upload_bytes(uri, buffer.getvalue())

    def save_json(self, uri: str, payload: dict[str, Any]) -> None:
        self.upload_bytes(
            uri,
            json.dumps(payload, indent=2).encode("utf-8"),
            content_type="application/json",
        )
