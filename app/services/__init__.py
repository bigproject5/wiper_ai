"""
Services module - 외부 서비스 연동 (S3, Kafka)
"""

import logging

# 패키지 초기화
logger = logging.getLogger(__name__)
logger.info("Services 패키지 초기화됨")

# 핵심 클래스들을 패키지 레벨에서 바로 import 가능하게 함
from .s3_io import S3Client
from .kafka_producer import kafka_producer

# 패키지에서 공개할 API 정의 (VideoProcessor 제거)
__all__ = [
    'S3Client',
    'kafka_producer'
]
