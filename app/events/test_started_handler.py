from typing import Dict, Any, Optional
from app.events.event_handler import BaseEventHandler
from app.events.event_dto import TestStartedEventDTO, AiDiagnosisCompletedEventDTO
from app.services.s3_io import S3Client
from app.services.kafka_producer import kafka_producer
from app.wiper_ai import WiperAI
from app.config import settings

class TestStartedEventHandler(BaseEventHandler):
    """TestStartedEvent 처리 핸들러 - 단순화됨"""

    def __init__(self):
        super().__init__()
        self.s3_client = S3Client()
        self.wiper_ai = WiperAI(settings.MODEL_PATH)

        # 모델 로딩
        if not self.wiper_ai.load_model():
            raise RuntimeError("와이퍼 AI 모델 로딩 실패")

    def get_event_type(self) -> str:
        return "TestStartedEvent"

    async def handle(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """TestStartedEvent 처리"""
        try:
            # 이벤트 파싱
            event = TestStartedEventDTO(**event_data)
            self.logger.info(f"검사 시작 - AuditID: {event.auditId}, Type: {event.inspectionType}")

            # WIPER 검사만 처리
            if event.inspectionType != "WIPER":
                self.logger.info(f"WIPER 검사가 아님 - 스킵")
                return None

            # S3에서 비디오 다운로드
            video_data = self._download_video(event.collectDataPath)
            if not video_data:
                raise ValueError("S3 다운로드 실패")

            # AI 진단 수행
            result = self.wiper_ai.predict(video_data)

            # 결과 발송
            self._send_result(event, result)

            self.logger.info(f"AI 진단 완료 - AuditID: {event.auditId}")
            return {"status": "success", "auditId": event.auditId}

        except Exception as e:
            self.logger.error(f"TestStartedEvent 처리 실패: {e}")
            self._send_error(event_data, str(e))
            return {"status": "error", "error": str(e)}

    def _download_video(self, s3_path: str) -> Optional[bytes]:
        """S3에서 비디오 다운로드"""
        try:
            self.logger.info(f"S3 다운로드 시작: {s3_path}")

            # S3 경로 파싱 개선
            if s3_path.startswith('s3://'):
                # s3://bucket/key 형태
                path_parts = s3_path.replace('s3://', '').split('/', 1)
                bucket, key = path_parts if len(path_parts) == 2 else (settings.S3_BUCKET, path_parts[0])
            elif s3_path.startswith('https://') and '.s3.amazonaws.com' in s3_path:
                # https://bucket.s3.amazonaws.com/key 형태
                # URL에서 버킷명과 키 추출
                import re
                match = re.match(r'https://([^.]+)\.s3\.amazonaws\.com/(.+)', s3_path)
                if match:
                    bucket, key = match.groups()
                    self.logger.info(f"HTTPS URL 파싱 결과 - Bucket: {bucket}, Key: {key}")
                else:
                    self.logger.error(f"HTTPS S3 URL 파싱 실패: {s3_path}")
                    return None
            else:
                # 키만 있는 경우
                bucket, key = settings.S3_BUCKET, s3_path

            self.logger.info(f"S3 다운로드 시도 - Bucket: {bucket}, Key: {key}")

            # 다운로드
            data = self.s3_client.download_data(bucket=bucket, key=key)
            self.logger.info(f"S3 다운로드 완료 - 크기: {len(data)} bytes")
            return data

        except Exception as e:
            self.logger.error(f"S3 다운로드 실패: {e}")
            return None

    def _send_result(self, event: TestStartedEventDTO, result: Dict[str, Any]):
        """결과 발송"""
        try:
            # 진단 결과에서 클래스명만 추출
            predicted_class = result.get('predicted_class', 'unknown')

            # AI 진단 완료 이벤트 DTO 생성
            ai_result = AiDiagnosisCompletedEventDTO(
                auditId=event.auditId,
                inspectionId=event.inspectionId,
                inspectionType=event.inspectionType,
                isDefect=result['is_defect'],
                collectDataPath=event.collectDataPath,
                resultDataPath=event.collectDataPath,  # collectDataPath 그대로 사용
                diagnosisResult=predicted_class  # 진단된 클래스명만
            )

            # Kafka로 결과 발송
            success = kafka_producer.send_ai_diagnosis_result(ai_result)
            if success:
                self.logger.info(f"AI 진단 결과 발송 성공 - 클래스: {predicted_class}, 결함여부: {result['is_defect']}")
                self.logger.info(f"발송된 데이터: {ai_result.dict()}")
            else:
                self.logger.error("AI 진단 결과 발송 실패")

        except Exception as e:
            self.logger.error(f"결과 발송 중 오류: {e}")

    def _send_error(self, event_data: Dict[str, Any], error_msg: str):
        """오류 결과 발송"""
        try:
            error_result = AiDiagnosisCompletedEventDTO(
                auditId=event_data.get('auditId', 0),
                inspectionId=event_data.get('inspectionId', 0),
                inspectionType=event_data.get('inspectionType', 'UNKNOWN'),
                isDefect=False,
                collectDataPath=event_data.get('collectDataPath', ''),
                resultDataPath='',
                diagnosisResult=f"처리 오류: {error_msg}"
            )
            kafka_producer.send_ai_diagnosis_result(error_result)
        except Exception as e:
            self.logger.error(f"오류 메시지 발송 실패: {e}")
