import os
from typing import Dict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """환경 설정 클래스"""
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Kafka 설정 - TestStartedEvent 구독용
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    KAFKA_INPUT_TOPIC: str = "test-started"  # TestStartedEvent 토픽
    KAFKA_OUTPUT_TOPIC: str = "ai-diagnosis-completed"  # AI 결과 발행용 (기존 유지)
    KAFKA_CONSUMER_GROUP: str = "ai-wiper-group"  # AI 서비스 전용 그룹

    # S3 설정
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = "ap-northeast-2"
    S3_BUCKET: str = "aivle-5"

    # 모델 설정
    MODEL_PATH: str = "models/wiper_model.pth"
    MODEL_VERSION: str = "1.0.0"
    MODEL_TYPE: str = "pytorch"

    # AI 추론 방법론 설정
    AGGREGATION_METHOD: str = "simple_majority"  # simple_majority, confidence_weighted, threshold_based, temporal_consistency, ensemble

    # 추론 방법별 세부 설정
    CONFIDENCE_THRESHOLD: float = 0.7  # threshold_based 방법용
    MIN_DEFECT_RATIO: float = 0.3      # threshold_based 방법용
    MIN_SEQUENCE_LENGTH: int = 3       # temporal_consistency 방법용

    # 새로운 후처리 파이프라인 설정
    INFERENCE_PIPELINE_CONFIG: Dict = {
        "type": "custom",  # 커스텀 파이프라인으로 변경

        # 간단한 파이프라인: 다수결 + 시간적 일관성만 사용
        "processors": [
            {
                "type": "simple_majority",
                "weight": 1.0,
                "params": {}
            },
            {
                "type": "temporal_consistency",
                "weight": 1.2,  # 시간적 일관성에 약간 더 높은 가중치
                "params": {
                    "min_sequence_length": 2,  # 연속 길이 조건을 낮춤 (2개 이상)
                    "coverage_weight": 0.3     # 커버리지 가중치 낮춤
                }
            }
        ]

        # 다른 설정들은 주석 처리 (사용하지 않음)
        # "confidence_threshold": 0.7,
        # "min_defect_ratio": 0.3,
        # "weight_power": 2.0,
    }

    class Config:
        env_file = ".env"


settings = Settings()
