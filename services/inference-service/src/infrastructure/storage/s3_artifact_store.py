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

    def load_json(self, uri: str) -> dict[str, Any]:
        payload = json.loads(self.download_bytes(uri).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"S3 JSON artifact must contain an object: {uri}")
        return payload

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

    def list_uris(self, bucket: str, prefix: str) -> list[str]:
        normalized_prefix = prefix.strip("/")
        paginator = self._client.get_paginator("list_objects_v2")
        uris: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/"):
                    continue
                uris.append(f"s3://{bucket}/{key}")
        return sorted(uris)

    def copy_uri(self, source_uri: str, destination_uri: str) -> None:
        source_bucket, source_key = parse_s3_uri(source_uri)
        destination_bucket, destination_key = parse_s3_uri(destination_uri)
        self.ensure_bucket(destination_bucket)
        self._client.copy(
            CopySource={"Bucket": source_bucket, "Key": source_key},
            Bucket=destination_bucket,
            Key=destination_key,
        )

    def delete_uri(self, uri: str) -> None:
        bucket, key = parse_s3_uri(uri)
        self._client.delete_object(Bucket=bucket, Key=key)

    def move_uri(self, source_uri: str, destination_uri: str) -> None:
        self.copy_uri(source_uri, destination_uri)
        self.delete_uri(source_uri)

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
