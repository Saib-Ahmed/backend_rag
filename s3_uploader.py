#!/usr/bin/env python3
"""
s3_uploader.py
Utility script and class to manage file uploads to RunPod S3-compatible network volume.
"""

import os
import sys
import logging
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("s3_uploader")

try:
    import boto3
except ImportError:
    import subprocess
    import sys
    logger.info("boto3/botocore library not found. Attempting to install dynamically...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "botocore"])
        import boto3
        logger.info("boto3/botocore successfully installed dynamically!")
    except Exception as e:
        boto3 = None
        logger.error(f"Failed to dynamically install boto3: {e}")

# Default RunPod S3 Configuration
DEFAULT_ENDPOINT_URL = "https://s3api-eu-ro-1.runpod.io"
DEFAULT_BUCKET_NAME = "3lbhg4d7qx"
DEFAULT_REGION_NAME = "eu-ro-1"


class RunPodS3Uploader:
    def __init__(
        self,
        endpoint_url: str = None,
        bucket_name: str = None,
        access_key: str = None,
        secret_key: str = None,
        region_name: str = None,
    ):
        if boto3 is None:
            raise RuntimeError("boto3 package is missing. Run 'pip install boto3'")

        self.endpoint_url = endpoint_url or os.environ.get("RUNPOD_S3_ENDPOINT_URL", DEFAULT_ENDPOINT_URL)
        self.bucket_name = bucket_name or os.environ.get("RUNPOD_S3_BUCKET_NAME", DEFAULT_BUCKET_NAME)
        self.region_name = region_name or os.environ.get("RUNPOD_S3_REGION", DEFAULT_REGION_NAME)

        self.access_key = access_key or os.environ.get("RUNPOD_S3_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("RUNPOD_S3_SECRET_KEY")

        if not self.access_key or not self.secret_key:
            logger.warning("RUNPOD_S3_ACCESS_KEY or RUNPOD_S3_SECRET_KEY is not set.")
            self.s3_client = None
            return

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name,
            config=Config(signature_version='s3v4', retries={'max_attempts': 3, 'mode': 'standard'})
        )

    def is_configured(self) -> bool:
        return self.s3_client is not None

    def upload_file(self, local_path: str, s3_key: str = None) -> bool:
        """Uploads a local file to the RunPod S3 network volume."""
        if not self.s3_client:
            logger.warning("S3 client not initialized. Skipping upload.")
            return False

        if not os.path.exists(local_path):
            logger.error(f"Local file does not exist: {local_path}")
            return False

        if not s3_key:
            s3_key = os.path.basename(local_path)

        try:
            logger.info(f"Uploading {local_path} -> s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading {local_path} to S3: {e}")
            return False

    def download_file_to_stream(self, s3_key: str):
        """Returns a binary stream of the S3 object if it exists."""
        if not self.s3_client:
            return None
        try:
            logger.info(f"Fetching from S3: s3://{self.bucket_name}/{s3_key}")
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body']
        except ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                logger.warning(f"Key {s3_key} does not exist in S3.")
                return None
            logger.error(f"S3 get_object error for {s3_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected S3 error for {s3_key}: {e}")
            return None


# Global helper instance
_uploader_instance = None

def get_s3_uploader() -> RunPodS3Uploader:
    global _uploader_instance
    if _uploader_instance is None:
        try:
            _uploader_instance = RunPodS3Uploader()
        except Exception as e:
            logger.warning(f"Could not initialize S3 uploader: {e}")
            _uploader_instance = None
    return _uploader_instance

def upload_pdf_to_s3(local_pdf_path: str, s3_key: str = None) -> bool:
    uploader = get_s3_uploader()
    if uploader and uploader.is_configured():
        return uploader.upload_file(local_pdf_path, s3_key)
    return False

def download_pdf_from_s3(s3_key: str):
    uploader = get_s3_uploader()
    if uploader and uploader.is_configured():
        return uploader.download_file_to_stream(s3_key)
    return None
