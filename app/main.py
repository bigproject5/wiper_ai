import asyncio
import json
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from kafka import KafkaConsumer
from threading import Thread

from app.config import settings
from app.schemas import (
    TestStartedEventDTO,
    AiDiagnosisCompletedEventDTO,
    HealthResponse,
    PredictionRequest,
    PredictionResponse
)
from app.s3_io import S3Client
from app.kafka_producer import kafka_producer
from app.inference import get_model, initialize_model

# 로깅 설정
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 전역 인스턴스
s3_client = S3Client()
# inference.py의 전역 인스턴스를 사용
wiper_model = get_model()


def kafka_consumer_thread():
    """TestStartedEvent Kafka 메시지를 소비하는 스레드"""
    logger.info(f"TestStartedEvent Kafka 컨슈머 시작 - Topic: {settings.KAFKA_INPUT_TOPIC}, Group: {settings.KAFKA_CONSUMER_GROUP}")

    # 수동 할당 전용 컨슈머 생성
    consumer = KafkaConsumer(
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id=settings.KAFKA_CONSUMER_GROUP,  # 컨슈머 그룹 설정 추가
        auto_offset_reset='latest',  # 최신 메시지부터 읽기로 변경
        enable_auto_commit=True,
        fetch_min_bytes=1,
        session_timeout_ms=30000,
        heartbeat_interval_ms=3000,
        max_poll_interval_ms=300000,
        request_timeout_ms=60000
    )

    # 토픽 파티션 정보 확인
    partitions = consumer.partitions_for_topic(settings.KAFKA_INPUT_TOPIC)
    logger.info(f"토픽 '{settings.KAFKA_INPUT_TOPIC}' 파티션: {partitions}")

    if not partitions:
        logger.error(f"토픽 '{settings.KAFKA_INPUT_TOPIC}'의 파티션을 찾을 수 없습니다")
        return

    # 바로 수동 파티션 할당
    from kafka import TopicPartition
    topic_partitions = [TopicPartition(settings.KAFKA_INPUT_TOPIC, p) for p in partitions]
    consumer.assign(topic_partitions)
    logger.info(f"수동으로 파티션 할당 완료: {topic_partitions}")

    # 메시지 소비 시작
    logger.info("메시지 수신 대기 중...")
    for message in consumer:
        try:
            logger.info(f"TestStartedEvent 메시지 수신: Partition={message.partition}, Offset={message.offset}")
            logger.info(f"메시지 내용: {message.value}")
            process_test_started_event(message.value)
        except Exception as e:
            logger.error(f"TestStartedEvent 메시지 처리 오류: {e}")


def process_test_started_event(message_data):
    """TestStartedEvent 메시지를 처리하고 AI 분석 수행"""
    try:
        # TestStartedEvent 메시지 파싱
        event = TestStartedEventDTO(**message_data)

        logger.info(f"검사 시작 - AuditID: {event.auditId}, Type: {event.inspectionType}, Model: {event.model}")

        # WIPER 검사 유형만 처리
        if event.inspectionType != "WIPER":
            logger.info(f"WIPER 검사가 아님 - InspectionType: {event.inspectionType}, 스킵")
            return

        # S3에서 데이터 다운로드
        logger.info(f"S3에서 데이터 다운로드 시작 - Path: {event.collectDataPath}")

        # S3 경로에서 버킷과 키 분리
        s3_path = event.collectDataPath
        
        if s3_path.startswith('s3://'):
            # s3://bucket-name/key 형식에서 파싱
            path_parts = s3_path.replace('s3://', '').split('/', 1)
            if len(path_parts) == 2:
                bucket_name, s3_key = path_parts
                logger.info(f"S3 경로 파싱 - Bucket: {bucket_name}, Key: {s3_key}")
            else:
                # 경로가 올바르지 않은 경우 기본 설정 사용
                bucket_name = settings.S3_BUCKET
                s3_key = path_parts[0] if path_parts else s3_path
                logger.warning(f"S3 경로 파싱 실패, 기본 설정 사용 - Bucket: {bucket_name}, Key: {s3_key}")
        elif s3_path.startswith('https://') and '.s3.amazonaws.com/' in s3_path:
            # https://bucket-name.s3.amazonaws.com/key 형식에서 파싱
            try:
                # URL에서 버킷명과 키 추출
                url_parts = s3_path.replace('https://', '').split('.s3.amazonaws.com/', 1)
                if len(url_parts) == 2:
                    bucket_name = url_parts[0]
                    s3_key = url_parts[1]
                    logger.info(f"HTTPS S3 URL 파싱 - Bucket: {bucket_name}, Key: {s3_key}")
                else:
                    raise ValueError("HTTPS URL 파싱 실패")
            except Exception as parse_error:
                logger.error(f"HTTPS S3 URL 파싱 오류: {parse_error}")
                # 폴백: 기본 버킷 사용하고 URL 전체를 키로 시도
                bucket_name = settings.S3_BUCKET
                s3_key = s3_path
                logger.warning(f"URL 파싱 실패, 기본 설정 사용 - Bucket: {bucket_name}, Key: {s3_key}")
        else:
            # s3:// 프로토콜이 없는 경우 키로 간주
            bucket_name = settings.S3_BUCKET
            s3_key = s3_path
            logger.info(f"S3 키로 처리 - Bucket: {bucket_name}, Key: {s3_key}")

        # S3에서 데이터 다운로드
        data = s3_client.download_data(
            bucket=bucket_name,
            key=s3_key
        )
        logger.info(f"S3 다운로드 성공 - Size: {len(data)} bytes")

        # AI 모델 분석 수행 (클립 저장 옵션 추가)
        start_time = time.time()
        prediction = wiper_model.predict(
            data,
            audit_id=str(event.auditId),
            save_clips=True  # 클립 저장 활성화
        )
        processing_time = time.time() - start_time

        # 새로운 inference.py의 결과 형식에 맞춰 수정
        is_defect = prediction.get('is_defect', False)
        predicted_class = prediction.get('predicted_class', 'unknown')
        confidence = prediction.get('confidence', 0.0)
        diagnosis_result = prediction.get('diagnosis_result', '진단 결과 없음')
        details = prediction.get('details', {})
        saved_clips = prediction.get('saved_clips', [])

        # 상세 진단 결과 생성 (기존 결과에 처리 시간 추가)
        diagnosis_detail = f"{diagnosis_result}, 처리시간: {processing_time:.3f}초"
        if saved_clips:
            diagnosis_detail += f", 저장된 클립 수: {len(saved_clips)}"

        if is_defect:
            diagnosis_detail += f" - 결함 감지됨 ({predicted_class})"

            # 결함 타입별 상세 설명 추가
            defect_descriptions = {
                'angle_limit': '와이퍼 각도 제한 문제',
                'left_fail': '좌측 와이퍼 동작 실패',
                'right_wiper_fail': '우측 와이퍼 동작 실패',
                'slow': '와이퍼 동작 속도 저하',
                'wiper_lag': '와이퍼 동작 지연',
                'wiper_stop': '와이퍼 동작 정지'
            }

            if predicted_class in defect_descriptions:
                diagnosis_detail += f" - {defect_descriptions[predicted_class]}"
        else:
            diagnosis_detail += " - 정상 상태"

        # 클래스별 확률 정보 추가
        class_probs = prediction.get('class_probabilities', {})
        if class_probs:
            top_3_classes = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            prob_info = ", ".join([f"{cls}: {prob:.3f}" for cls, prob in top_3_classes])
            diagnosis_detail += f" (상위3개: {prob_info})"

        # 결과 데이터 S3 경로 생성 (원본 경로 기반)
        result_data_path = event.collectDataPath.replace('.mp4', '_result.json').replace('.jpg', '_result.json').replace('.png', '_result.json')

        # AI 진단 완료 결과 메시지 생성
        ai_result = AiDiagnosisCompletedEventDTO(
            auditId=event.auditId,
            inspectionId=event.inspectionId,
            inspectionType=event.inspectionType,
            isDefect=is_defect,
            collectDataPath=event.collectDataPath,
            resultDataPath=result_data_path,
            diagnosisResult=diagnosis_detail
        )

        # Kafka로 AI 진단 결과 발행
        success = kafka_producer.send_ai_diagnosis_result(ai_result)

        if success:
            logger.info(f"AI 진단 완료 - AuditID: {event.auditId}, 결함여부: {is_defect}, 결과: {predicted_class}")
        else:
            logger.error(f"AI 결과 발행 실패 - AuditID: {event.auditId}")

    except Exception as e:
        logger.error(f"TestStartedEvent 처리 실패: {e}")

        # 오류 발생 시 오류 메시지도 Kafka로 발행
        try:
            error_result = AiDiagnosisCompletedEventDTO(
                auditId=message_data.get('auditId', 0),
                inspectionId=message_data.get('inspectionId', 0),
                inspectionType=message_data.get('inspectionType', 'UNKNOWN'),
                isDefect=False,
                collectDataPath=message_data.get('collectDataPath', ''),
                resultDataPath='',
                diagnosisResult=f"처리 오류: {str(e)}"
            )
            kafka_producer.send_ai_diagnosis_result(error_result)
        except Exception as error_e:
            logger.error(f"오류 메시지 발행 실패: {error_e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 함수"""
    # 시작 시
    logger.info("Wiper AI 서버 시작")

    # 모델 초기화 및 로딩
    logger.info("모델 로딩 시작...")
    model_loaded = initialize_model()
    logger.info(f"모델 로드 상태: {model_loaded}")

    if not model_loaded:
        logger.error("모델 로딩 실패! 서버 시작을 중단합니다.")
        raise RuntimeError("모델 로딩 실패")

    # 전역 wiper_model 인스턴스의 model_loaded 상태도 업데이트
    wiper_model.model_loaded = model_loaded
    logger.info(f"전역 모델 인스턴스 상태 업데이트: {wiper_model.model_loaded}")

    # Kafka 컨슈머 스레드 시작
    consumer_thread = Thread(target=kafka_consumer_thread, daemon=True)
    consumer_thread.start()
    logger.info("Kafka 컨슈머 스레드 시작됨")

    yield

    # 종료 시
    logger.info("Wiper AI 서버 종료")
    kafka_producer.close()


# FastAPI 앱 생성
app = FastAPI(
    title="Wiper AI API",
    description="와이퍼 이상 탐지 AI API 서버 - TestStartedEvent 구독",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=wiper_model.model_loaded,
        kafka_connected=True
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """직접 예측 요청 (테스트용)"""
    try:
        start_time = time.time()

        # S3에서 데이터 다운로드 (선택적)
        if request.s3_bucket and request.s3_key:
            data = s3_client.download_data(
                bucket=request.s3_bucket,
                key=request.s3_key
            )
        else:
            data = request.data

        # 모델 예측
        result = wiper_model.predict(data)

        return PredictionResponse(
            result=result,
            confidence=result.get('confidence', 0.0),
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"예측 요청 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
