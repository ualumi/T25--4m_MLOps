"""S3-совместимое хранилище загружаемых датасетов."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError


class S3DatasetStore:
    def __init__(
        self,
        bucket: str,
        prefix: str,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._region = region
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
        )

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code not in {"404", "NoSuchBucket"}:
                raise

            create_kwargs: dict[str, object] = {"Bucket": self._bucket}
            if self._region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._region
                }
            self._client.create_bucket(**create_kwargs)

    def save_dataset(
        self,
        user_id: str,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        self.ensure_bucket()
        safe_filename = self._sanitize_filename(filename)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        key = "/".join(
            part
            for part in (
                self._prefix,
                user_id,
                f"{timestamp}-{uuid4().hex}-{safe_filename}",
            )
            if part
        )
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return f"s3://{self._bucket}/{key}"

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        original_name = Path(filename).name or "dataset"
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
        return sanitized or "dataset"
