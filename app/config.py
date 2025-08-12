import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """환경 설정 클래스"""
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Kafka 설정 - TestStartedEvent 구독용
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_INPUT_TOPIC: str = "test-started"  # TestStartedEvent 토픽
    KAFKA_OUTPUT_TOPIC: str = "ai-diagnosis-completed"  # AI 결과 발행용 (기존 유지)
    KAFKA_CONSUMER_GROUP: str = "ai-wiper-group"  # AI 서비스 전용 그룹

    # S3 설정
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = "ap-southeast-2"
    S3_BUCKET: str = "aivle-5"

    # 모델 설정
    MODEL_PATH: str = "models/wiper_model.pth"
    MODEL_VERSION: str = "1.0.0"
    MODEL_TYPE: str = "pytorch"

    class Config:
        env_file = ".env"


settings = Settings()
