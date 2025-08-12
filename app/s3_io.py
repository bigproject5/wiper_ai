import boto3
import logging
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """S3 클라이언트 클래스"""

    def __init__(self):
        self.s3_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """S3 클라이언트 초기화"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            logger.info("S3 클라이언트 초기화 완료")
        except NoCredentialsError:
            logger.warning("AWS 자격 증명이 설정되지 않았습니다")
        except Exception as e:
            logger.error(f"S3 클라이언트 초기화 실패: {e}")
    
    def download_data(self, bucket: str, key: str) -> bytes:
        """S3에서 데이터를 다운로드"""
        try:
            if not self.s3_client:
                raise RuntimeError("S3 클라이언트가 초기화되지 않았습니다")

            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()

            logger.info(f"S3 다운로드 완료: s3://{bucket}/{key}")
            return data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"S3 다운로드 실패 ({error_code}): {e}")
            raise
        except Exception as e:
            logger.error(f"S3 다운로드 실패: {e}")
            raise

    def upload_data(self, bucket: str, key: str, data: bytes) -> bool:
        """S3에 데이터를 업로드"""
        try:
            if not self.s3_client:
                raise RuntimeError("S3 클라이언트가 초기화되지 않았습니다")

            self.s3_client.put_object(Bucket=bucket, Key=key, Body=data)
            logger.info(f"S3 업로드 완료: s3://{bucket}/{key}")
            return True
            
        except ClientError as e:
            logger.error(f"S3 업로드 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"S3 업로드 실패: {e}")
            return False
    
    def check_object_exists(self, bucket: str, key: str) -> bool:
        """S3 객체 존재 여부 확인"""
        try:
            if not self.s3_client:
                return False

            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"S3 객체 확인 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"S3 객체 확인 실패: {e}")
            return False
