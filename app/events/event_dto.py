from pydantic import BaseModel


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
