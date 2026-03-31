"""S3-совместимое хранилище файлов и JSON-артефактов."""

from __future__ import annotations

import json
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
        subfolder: str | None = None,
    ) -> str:
        self.ensure_bucket()
        safe_filename = self._sanitize_filename(filename)
        return self.save_bytes(
            user_id=user_id,
            filename=safe_filename,
            data=data,
            content_type=content_type,
            subfolder=subfolder,
        )

    def save_json_artifact(
        self,
        user_id: str,
        name: str,
        payload: object,
        subfolder: str | None = None,
    ) -> str:
        safe_name = self._sanitize_filename(name)
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return self.save_bytes(
            user_id=user_id,
            filename=safe_name,
            data=data,
            content_type="application/json",
            subfolder=subfolder,
        )

    def save_bytes(
        self,
        user_id: str,
        filename: str,
        data: bytes,
        content_type: str,
        subfolder: str | None = None,
    ) -> str:
        self.ensure_bucket()
        key = self._build_key(user_id=user_id, filename=filename, subfolder=subfolder)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return f"s3://{self._bucket}/{key}"

    def _build_key(
        self,
        user_id: str,
        filename: str,
        subfolder: str | None = None,
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return "/".join(
            part
            for part in (
                self._prefix,
                user_id,
                (subfolder or "").strip("/"),
                f"{timestamp}-{uuid4().hex}-{filename}",
            )
            if part
        )

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        original_name = Path(filename).name or "dataset"
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
        return sanitized or "dataset"
