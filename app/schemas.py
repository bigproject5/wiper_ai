from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class WiperInputMessage(BaseModel):
    """Kafka 입력 메시지 스키마"""
    request_id: str
    s3_bucket: str
    s3_key: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None

class WiperOutputMessage(BaseModel):
    """Kafka 출력 메시지 스키마"""
    request_id: str
    result: str
    confidence: float
    processing_time: float
    timestamp: datetime
    model_version: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str
    timestamp: datetime
    model_loaded: bool
    kafka_connected: bool

class PredictionRequest(BaseModel):
    """직접 예측 요청 스키마"""
    data: Optional[Any] = None
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None

class PredictionResponse(BaseModel):
    """예측 응답 스키마"""
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: datetime

class TestStartedEventDTO(BaseModel):
    """TestStartedEvent Kafka 메시지 스키마"""
    auditId: int  # 검사 ID
    model: str  # 차량 모델명 (예: sonata)
    lineCode: str  # 라인 코드 (예: A2)
    inspectionId: int  # 검사 항목 ID
    inspectionType: str  # 검사 유형 (PAINT, LAMP, WIPER 등)
    collectDataPath: str  # S3 파일 경로

class AiDiagnosisCompletedEventDTO(BaseModel):
    """AI 진단 완료 이벤트 DTO"""
    auditId: int  # 검사 ID
    inspectionId: int  # 검사 항목 ID
    inspectionType: str  # 검사 유형 (PAINT, LAMP, WIPER 등)
    isDefect: bool  # 결함 여부 (True: 결함, False: 정상)
    collectDataPath: str  # 원본 데이터 S3 경로
    resultDataPath: str  # 결과 데이터 S3 경로
    diagnosisResult: str  # 상세 진단 결과 설명
