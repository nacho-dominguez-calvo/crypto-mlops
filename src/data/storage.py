# src/data/storage.py
import boto3
import os
import logging
from botocore.exceptions import ClientError
from typing import Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class S3Storage:
    def __init__(self, bucket_name: str, region_name: Optional[str] = None):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            region_name=region_name or os.getenv("AWS_REGION")
        )

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            return False

    def save_bytes(self, data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> bool:
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=content_type,
            )
            logger.info(f"Saved object {s3_key} to s3://{self.bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Save failed: {e}")
            return False

    def read_bytes(self, s3_key: str) -> Optional[bytes]:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Read object {s3_key} from s3://{self.bucket_name}")
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Read failed: {e}")
            return None

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if "Contents" in response:
                keys = [item["Key"] for item in response["Contents"]]
                logger.info(f"Listed {len(keys)} files with prefix '{prefix}' in {self.bucket_name}")
                return keys
            logger.info(f"No files found with prefix '{prefix}' in {self.bucket_name}")
            return []
        except ClientError as e:
            logger.error(f"List failed: {e}")
            return []
    def write_bytes(self, s3_key: str, data: bytes) -> None:
        """
        Write raw bytes to S3 at the given key.
        """
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=s3_key, Body=data)
            logger.info(f"Saved object {s3_key} to s3://{self.bucket_name}")
        except Exception as e:
            logger.error(f"Write failed for {s3_key}: {e}")
            raise

