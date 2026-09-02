import os
import io
import logging
from typing import Optional, Generator
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("s3_service")

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 is not installed. S3 features will be disabled.")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "lexai-documents-storage-2026")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

_s3_client = None

def get_s3_client():
    global _s3_client
    if not BOTO3_AVAILABLE:
        return None
    if _s3_client is None:
        try:
            # First check if explicit credentials exist in environment
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
            if access_key and secret_key:
                _s3_client = boto3.client(
                    "s3",
                    region_name=AWS_REGION,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key
                )
            else:
                # Use EC2 IAM Role automatically
                _s3_client = boto3.client("s3", region_name=AWS_REGION)
                
            logger.info(f"Initialized S3 client for bucket '{S3_BUCKET_NAME}' in region '{AWS_REGION}'")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            _s3_client = None
    return _s3_client

def upload_file_to_s3(local_path: str, s3_key: str, content_type: Optional[str] = None) -> bool:
    """Upload a local file to S3 bucket."""
    client = get_s3_client()
    if not client or not S3_BUCKET_NAME:
        logger.warning("S3 client not available or S3_BUCKET_NAME not set.")
        return False
    
    if not os.path.exists(local_path):
        logger.error(f"Local file does not exist: {local_path}")
        return False

    try:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        elif local_path.lower().endswith(".pdf"):
            extra_args["ContentType"] = "application/pdf"
        elif local_path.lower().endswith(".md"):
            extra_args["ContentType"] = "text/markdown"
            
        client.upload_file(local_path, S3_BUCKET_NAME, s3_key, ExtraArgs=extra_args if extra_args else None)
        logger.info(f"Successfully uploaded '{local_path}' to 's3://{S3_BUCKET_NAME}/{s3_key}'")
        return True
    except Exception as e:
        logger.error(f"Failed to upload '{local_path}' to S3: {e}")
        return False

def upload_bytes_to_s3(file_bytes: bytes, s3_key: str, content_type: str = "application/octet-stream") -> bool:
    """Upload in-memory bytes directly to S3."""
    client = get_s3_client()
    if not client or not S3_BUCKET_NAME:
        return False
    try:
        client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes,
            ContentType=content_type
        )
        logger.info(f"Successfully uploaded {len(file_bytes)} bytes to 's3://{S3_BUCKET_NAME}/{s3_key}'")
        return True
    except Exception as e:
        logger.error(f"Failed to upload bytes to S3: {e}")
        return False

def download_file_from_s3(s3_key: str, local_destination_path: str) -> bool:
    """Download an S3 object to local disk."""
    client = get_s3_client()
    if not client or not S3_BUCKET_NAME:
        return False
    try:
        os.makedirs(os.path.dirname(local_destination_path), exist_ok=True)
        client.download_file(S3_BUCKET_NAME, s3_key, local_destination_path)
        logger.info(f"Downloaded 's3://{S3_BUCKET_NAME}/{s3_key}' to '{local_destination_path}'")
        return True
    except Exception as e:
        logger.error(f"Failed to download 's3://{S3_BUCKET_NAME}/{s3_key}': {e}")
        return False

def get_s3_presigned_url(s3_key: str, expiration: int = 3600) -> Optional[str]:
    """Generate a pre-signed URL for direct frontend client download/viewing."""
    client = get_s3_client()
    if not client or not S3_BUCKET_NAME:
        return None
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL for '{s3_key}': {e}")
        return None

def stream_s3_file_bytes(s3_key: str) -> Optional[bytes]:
    """Fetch complete object bytes from S3 into memory."""
    client = get_s3_client()
    if not client or not S3_BUCKET_NAME:
        return None
    try:
        response = client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return response["Body"].read()
    except Exception as e:
        logger.error(f"Failed to stream S3 bytes for '{s3_key}': {e}")
        return None

def check_s3_file_exists(s3_key: str) -> bool:
    """Check if object exists in S3."""
    client = get_s3_client()
    if not client or not S3_BUCKET_NAME:
        return False
    try:
        client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        logger.error(f"Error checking S3 file existence for '{s3_key}': {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking S3 file existence for '{s3_key}': {e}")
        return False
