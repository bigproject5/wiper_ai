import asyncio
import json
import logging
from datetime import datetime
from threading import Thread
from contextlib import asynccontextmanager
from fastapi import FastAPI
from kafka import KafkaConsumer
from kafka.structs import TopicPartition

from app.config import settings
from app.events.event_router import EventRouter

# 로깅 설정
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 전역 인스턴스
event_router = EventRouter()


def kafka_consumer_thread():
    """Kafka 메시지 소비 스레드 - Consumer Group 방식"""
    logger.info(f"Kafka 컨슈머 시작 - Topic: {settings.KAFKA_INPUT_TOPIC}")

    try:
        # Consumer Group 방식으로 토픽 구독
        consumer = KafkaConsumer(
            settings.KAFKA_INPUT_TOPIC,  # 토픽을 직접 구독
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=settings.KAFKA_CONSUMER_GROUP,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            session_timeout_ms=10000,
            heartbeat_interval_ms=3000,
            max_poll_interval_ms=300000,
            fetch_min_bytes=1,
            fetch_max_wait_ms=500
        )

        logger.info("Consumer Group 방식으로 파티션 할당 대기 중...")

        # 파티션 할당 대기 (Consumer Group이 자동으로 할당)
        max_wait_time = 30
        wait_count = 0

        while wait_count < max_wait_time:
            # poll()을 호출해야 파티션 할당이 시작됨
            message_batch = consumer.poll(timeout_ms=1000)
            assignments = consumer.assignment()

            if assignments:
                logger.info(f"파티션 할당 완료: {assignments}")
                break
            else:
                logger.info(f"파티션 할당 대기 중... ({wait_count + 1}/{max_wait_time}초)")
                wait_count += 1

        if not consumer.assignment():
            logger.error("파티션 할당 실패 - Consumer 종료")
            consumer.close()
            return

        logger.info("메시지 수신 대기 중...")

        # 메시지 처리 루프
        for message in consumer:
            try:
                logger.info(f"메시지 수신 - Partition: {message.partition}, Offset: {message.offset}")
                logger.info(f"메시지 내용: {message.value}")

                # 이벤트 처리
                result = asyncio.run(event_router.route_event("TestStartedEvent", message.value))

                if result and result.get('status') == 'success':
                    logger.info(f"이벤트 처리 성공: {result}")
                elif result and result.get('status') == 'error':
                    logger.error(f"이벤트 처리 실패: {result.get('error')}")

            except Exception as e:
                logger.error(f"메시지 처리 오류: {e}")
                continue

    except Exception as e:
        logger.error(f"Kafka Consumer 오류: {e}")
        # 5초 후 재시작
        import time
        time.sleep(5)
        kafka_consumer_thread()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료"""
    logger.info("Wiper AI 서버 시작")

    # Kafka 컨슈머 스레드 시작
    consumer_thread = Thread(target=kafka_consumer_thread, daemon=True)
    consumer_thread.start()

    yield

    logger.info("Wiper AI 서버 종료")


# FastAPI 앱
app = FastAPI(
    title="Wiper AI",
    description="와이퍼 이상 탐지 AI 서버",
    version="3.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": True,
        "kafka_connected": True,
        "supported_events": ["TestStartedEvent"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
