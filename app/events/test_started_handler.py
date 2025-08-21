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

        # 새로운 후처리 파이프라인 설정으로 WiperAI 초기화
        self.wiper_ai = WiperAI(
            model_path=settings.MODEL_PATH,
            pipeline_config=settings.INFERENCE_PIPELINE_CONFIG
        )

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
        """결과 발송 - 종합적인 진단 결과 문자열 생성"""
        try:
            # 결과에서 필요한 정보 추출
            predicted_class = result.get('predicted_class', 'unknown')
            is_defect = result.get('is_defect', False)
            confidence = result.get('confidence', 0.0)
            total_clips = result.get('total_clips', 0)
            successful_clips = result.get('successful_clips', 0)
            processing_time = result.get('processing_time', 0.0)

            # 후처리 방법 정보 추출
            method_info = result.get('method_info', {})
            method_name = method_info.get('method', 'unknown')

            # 종합적인 진단 결과 문자열 생성
            diagnosis_result = self._generate_comprehensive_diagnosis(
                predicted_class=predicted_class,
                is_defect=is_defect,
                confidence=confidence,
                total_clips=total_clips,
                successful_clips=successful_clips,
                processing_time=processing_time,
                method_name=method_name,
                method_info=method_info,
                video_duration=total_clips * 2.0  # 2초 클립 기준
            )

            # AI 진단 완료 이벤트 DTO 생성
            ai_result = AiDiagnosisCompletedEventDTO(
                auditId=event.auditId,
                inspectionId=event.inspectionId,
                inspectionType=event.inspectionType,
                isDefect=is_defect,
                collectDataPath=event.collectDataPath,
                resultDataPath=event.collectDataPath,
                diagnosisResult=diagnosis_result  # 종합 진단 문자열
            )

            # Kafka로 결과 발송
            success = kafka_producer.send_ai_diagnosis_result(ai_result)
            if success:
                self.logger.info(f"AI 진단 결과 발송 성공")
                self.logger.info(f"- 클래스: {predicted_class}")
                self.logger.info(f"- 결함여부: {is_defect}")
                self.logger.info(f"- 후처리 방법: {method_name}")
                self.logger.info(f"- 처리 시간: {processing_time:.3f}초")
                self.logger.info(f"- 종합 진단: {diagnosis_result[:100]}...")  # 처음 100자만 로그
            else:
                self.logger.error("AI 진단 결과 발송 실패")

        except Exception as e:
            self.logger.error(f"결과 발송 중 오류: {e}")

    def _generate_comprehensive_diagnosis(self, predicted_class: str, is_defect: bool, confidence: float,
                                        total_clips: int, successful_clips: int, processing_time: float,
                                        method_name: str, method_info: Dict[str, Any], video_duration: float) -> str:
        """종합적인 진단 결과 문자열 생성 (255자 제한을 최대한 활용)"""

        # 1. 기본 영상 분석 정보
        video_info = f"{video_duration:.0f}초영상 {total_clips}클립분석({successful_clips}성공)"

        # 2. 와이퍼 상태 및 결함 여부
        status = "결함감지" if is_defect else "정상동작"
        wiper_result = f"와이퍼상태:{predicted_class}({status})"

        # 3. 후처리 전략 상세 정보 (한글로 변경)
        if method_name == "ensemble":
            participating_processors = method_info.get('participating_processors', [])
            individual_results = method_info.get('individual_results', {})
            processor_weights = method_info.get('processor_weights', {})

            # 개별 결과 요약 (한글 변환)
            result_summary = []
            for processor, result_class in individual_results.items():
                weight = processor_weights.get(processor, 1.0)
                # 영어 알고리즘명을 한글로 변환
                korean_name = self._get_korean_processor_name(processor)
                result_summary.append(f"{korean_name}({weight:.1f})→{result_class}")

            strategy_info = f"종합판정[{','.join(result_summary)}]"

        elif method_name == "simple_majority":
            vote_counts = method_info.get('vote_counts', {})
            vote_ratio = method_info.get('vote_ratio', 0.0)

            # 득표 현황
            vote_summary = [f"{cls}:{count}" for cls, count in sorted(vote_counts.items(), key=lambda x: -x[1])[:3]]
            strategy_info = f"다수결[득표:{','.join(vote_summary)} 최다:{vote_ratio:.1%}]"

        elif method_name == "temporal_consistency":
            class_sequences = method_info.get('class_sequences', {})
            min_sequence_length = method_info.get('min_sequence_length', 3)

            if predicted_class in class_sequences:
                seq_info = class_sequences[predicted_class]
                max_seq_len = seq_info.get('max_sequence_length', 0)
                coverage = seq_info.get('coverage', 0.0)
                sequence_count = seq_info.get('sequence_count', 0)
                strategy_info = f"시간패턴[최소길이:{min_sequence_length} 최대연속:{max_seq_len} 커버리지:{coverage:.1%} 패턴:{sequence_count}개]"
            else:
                strategy_info = f"시간패턴[최소길이:{min_sequence_length}]"

        elif method_name == "confidence_weighted":
            weighted_score = method_info.get('weighted_score', 0.0)
            weight_power = method_info.get('weight_power', 2.0)
            class_scores = method_info.get('class_scores', {})

            # 상위 클래스 점수
            top_scores = sorted(class_scores.items(), key=lambda x: -x[1])[:2]
            score_summary = [f"{cls}:{score:.2f}" for cls, score in top_scores]
            strategy_info = f"신뢰도가중[지수:{weight_power} 점수:{','.join(score_summary)}]"

        elif method_name == "threshold_based":
            confidence_threshold = method_info.get('confidence_threshold', 0.7)
            defect_ratio = method_info.get('defect_ratio', 0.0)
            min_defect_ratio = method_info.get('min_defect_ratio', 0.3)
            high_confidence_defects = method_info.get('high_confidence_defects', 0)

            strategy_info = f"임계값필터[신뢰도기준:{confidence_threshold} 결함비율:{defect_ratio:.1%}/{min_defect_ratio:.1%} 고신뢰결함:{high_confidence_defects}개]"

        else:
            strategy_info = f"{method_name}방식"

        # 4. 최종 성능 지표
        performance_info = f"신뢰도:{confidence:.3f} 처리시간:{processing_time:.2f}초"

        # 5. 모든 정보 조합
        full_result = f"{video_info} | {wiper_result} | {strategy_info} | {performance_info}"

        # 6. 255자 제한 처리 (스마트하게)
        if len(full_result) <= 255:
            return full_result

        # 초과 시 단계적 축약
        # Level 1: 후처리 전략 정보 축약
        strategy_short = self._get_short_strategy_info(method_name, method_info)
        level1_result = f"{video_info} | {wiper_result} | {strategy_short} | {performance_info}"

        if len(level1_result) <= 255:
            return level1_result

        # Level 2: 비디오 정보 축약
        video_short = f"{total_clips}클립({successful_clips}성공)"
        level2_result = f"{video_short} | {wiper_result} | {strategy_short} | {performance_info}"

        if len(level2_result) <= 255:
            return level2_result

        # Level 3: 최소 필수 정보만
        essential_result = f"{video_short} {status} {predicted_class} {strategy_short} 신뢰도{confidence:.3f}"

        return essential_result[:255]

    def _get_korean_processor_name(self, processor_name: str) -> str:
        """영어 후처리기 이름을 한글로 변환"""
        korean_names = {
            'simple_majority': '다수결',
            'confidence_weighted': '신뢰도가중',
            'threshold_based': '임계값필터',
            'temporal_consistency': '시간패턴'
        }
        return korean_names.get(processor_name, processor_name)

    def _get_short_strategy_info(self, method_name: str, method_info: Dict[str, Any]) -> str:
        """후처리 전략 정보 축약 버전 (한글)"""
        if method_name == "ensemble":
            participating_processors = method_info.get('participating_processors', [])
            return f"종합판정({len(participating_processors)}개알고리즘)"

        elif method_name == "simple_majority":
            vote_ratio = method_info.get('vote_ratio', 0.0)
            return f"다수결(득표율{vote_ratio:.1%})"

        elif method_name == "temporal_consistency":
            if method_info.get('class_sequences', {}):
                max_seq = max([seq.get('max_sequence_length', 0)
                              for seq in method_info['class_sequences'].values()])
                return f"시간패턴(최대연속{max_seq})"
            return "시간패턴"

        elif method_name == "confidence_weighted":
            weighted_score = method_info.get('weighted_score', 0.0)
            return f"신뢰도가중(점수{weighted_score:.2f})"

        elif method_name == "threshold_based":
            defect_ratio = method_info.get('defect_ratio', 0.0)
            return f"임계값필터(결함{defect_ratio:.1%})"

        else:
            return method_name

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
