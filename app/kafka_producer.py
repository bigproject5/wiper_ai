import json
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError

from app.config import settings
from app.schemas import AiDiagnosisCompletedEventDTO

logger = logging.getLogger(__name__)


class WiperKafkaProducer:
    """와이퍼 AI 결과를 Kafka로 발행하는 프로듀서"""

    def __init__(self):
        self.producer = None
        self._initialize_producer()

    def _initialize_producer(self):
        """Kafka 프로듀서 초기화"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                acks='all',  # 모든 복제본에서 확인
                retries=3,   # 재시도 횟수
                retry_backoff_ms=1000
            )
            logger.info("Kafka 프로듀서 초기화 완료")
        except Exception as e:
            logger.error(f"Kafka 프로듀서 초기화 실패: {e}")
            self.producer = None

    def send_ai_diagnosis_result(self, result: AiDiagnosisCompletedEventDTO):
        """AI 진단 완료 결과를 Kafka로 발행"""
        if not self.producer:
            logger.error("Kafka 프로듀서가 초기화되지 않았습니다")
            return False

        try:
            # Pydantic 모델을 딕셔너리로 변환
            message_data = result.dict()

            # Kafka로 메시지 발행
            future = self.producer.send(
                settings.KAFKA_OUTPUT_TOPIC,
                value=message_data
            )

            # 결과 확인 (비동기)
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)

            # 즉시 플러시 (동기적으로 전송 보장)
            self.producer.flush()

            logger.info(f"AI 진단 결과 발행 완료 - AuditID: {result.auditId}, InspectionID: {result.inspectionId}, 결함여부: {result.isDefect}")
            return True

        except KafkaError as e:
            logger.error(f"Kafka 메시지 발행 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"메시지 발행 중 오류: {e}")
            return False

    def _on_send_success(self, record_metadata):
        """메시지 전송 성공 콜백"""
        logger.debug(f"메시지 전송 성공 - Topic: {record_metadata.topic}, "
                    f"Partition: {record_metadata.partition}, "
                    f"Offset: {record_metadata.offset}")

    def _on_send_error(self, exception):
        """메시지 전송 실패 콜백"""
        logger.error(f"메시지 전송 실패: {exception}")

    def close(self):
        """프로듀서 종료"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka 프로듀서 종료")


# 전역 프로듀서 인스턴스
kafka_producer = WiperKafkaProducer()
