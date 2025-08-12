import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import os
from transformers import VivitImageProcessor, VivitForVideoClassification

from app.config import settings

logger = logging.getLogger(__name__)

class WiperModel:
    """ViViT 기반 와이퍼 진단 모델"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.model_loaded = False

        # ViViT 모델 클래스 매핑 (7개 클래스)
        self.class_mapping = {
            0: 'angle_limit',
            1: 'left_fail',
            2: 'normal',
            3: 'right_wiper_fail',
            4: 'slow',
            5: 'wiper_lag',
            6: 'wiper_stop'
        }

        logger.info(f"디바이스: {self.device}")

    def load_model(self, model_path: str) -> bool:
        """ViViT 모델 로딩"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
                return False

            logger.info(f"ViViT 모델 로딩 시작: {model_path}")

            # ViViT 프로세서 초기화 (먼저 시도)
            try:
                logger.info("ViViT 프로세서 로딩 시작...")
                self.processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
                logger.info("ViViT 프로세서 로딩 완료")
            except Exception as proc_error:
                logger.error(f"프로세서 로딩 실패: {proc_error}")
                return False

            # ViViT 모델 초기화
            try:
                logger.info("ViViT 모델 로딩 시작...")
                self.model = VivitForVideoClassification.from_pretrained(
                    "google/vivit-b-16x2-kinetics400",
                    num_labels=7,  # 7개 클래스
                    ignore_mismatched_sizes=True
                )
                logger.info("ViViT 기본 모델 로딩 완료")
            except Exception as model_error:
                logger.error(f"기본 모델 로딩 실패: {model_error}")
                return False

            # 파인튜닝된 가중치 로딩
            try:
                logger.info("파인튜닝된 가중치 로딩 시작...")
                checkpoint = torch.load(model_path, map_location=self.device)

                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    # 가중치 적용
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

                    if missing_keys:
                        logger.warning(f"누락된 키: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"예상치 못한 키: {unexpected_keys}")

                    logger.info("파인튜닝된 가중치 적용 완료")
                else:
                    # 완전한 모델 객체인 경우
                    self.model = checkpoint
                    logger.info("완전한 모델 객체 로딩 완료")
            except Exception as weight_error:
                logger.error(f"가중치 로딩 실패: {weight_error}")
                return False

            # 모델을 평가 모드로 설정하고 디바이스로 이동
            try:
                self.model.eval()
                self.model.to(self.device)
                logger.info(f"모델을 {self.device}로 이동 완료")
            except Exception as device_error:
                logger.error(f"디바이스 이동 실패: {device_error}")
                return False

            # 최종 검증
            if self.processor is None:
                logger.error("프로세서가 None입니다")
                return False

            if self.model is None:
                logger.error("모델이 None입니다")
                return False

            self.model_loaded = True
            logger.info(f"ViViT 모델 로딩 완료: {model_path}")
            logger.info(f"프로세서 상태: {type(self.processor).__name__}")
            logger.info(f"모델 상태: {type(self.model).__name__}")
            logger.info(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")

            return True

        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            self.model_loaded = False
            self.processor = None
            self.model = None
            return False

    def preprocess_video(self, video_bytes: bytes) -> List[List[np.ndarray]]:
        """ViViT 모델용 비디오 전처리"""
        tmp_file_path = None
        try:
            # 임시 파일 생성 (Windows 호환성을 위해 컨텍스트 매니저 밖에서 처리)
            tmp_fd, tmp_file_path = tempfile.mkstemp(suffix='.mp4')

            # 파일 디스크립터를 통해 데이터 쓰기
            with os.fdopen(tmp_fd, 'wb') as tmp_file:
                tmp_file.write(video_bytes)
                tmp_file.flush()

            # VideoCapture 열기
            cap = cv2.VideoCapture(tmp_file_path)

            if not cap.isOpened():
                raise ValueError("비디오 파일을 열 수 없습니다")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            logger.info(f"비디오 정보 - FPS: {fps}, 총 프레임: {total_frames}, 길이: {duration:.2f}초")

            # 2초 클립으로 슬라이딩 윈도우 (1초 stride)
            clip_duration = 2.0
            stride = 1.0

            clips_list = []
            start_time = 0.0

            while start_time + clip_duration <= duration:
                clip_frames = self._extract_frames_for_vivit(cap, start_time, clip_duration, fps)
                if clip_frames is not None:
                    clips_list.append(clip_frames)
                start_time += stride

            # VideoCapture 완전히 해제
            cap.release()

            # Windows에서는 cv2.destroyAllWindows()도 호출
            cv2.destroyAllWindows()

            if not clips_list:
                raise ValueError("비디오에서 클립을 추출할 수 없습니다")

            logger.info(f"추출된 클립 수: {len(clips_list)}")
            return clips_list

        except Exception as e:
            logger.error(f"비디오 전처리 실패: {e}")
            raise
        finally:
            # 임시 파일 정리 (에러가 발생해도 실행)
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    # 약간의 지연을 주어 파일 핸들이 완전히 해제되도록 함
                    import time
                    time.sleep(0.1)
                    os.unlink(tmp_file_path)
                    logger.debug(f"임시 파일 삭제 완료: {tmp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"임시 파일 삭제 실패: {cleanup_error}, 경로: {tmp_file_path}")
                    # Windows에서는 때로는 다음 GC 사이클에서 정리됨

    def _extract_frames_for_vivit(self, cap, start_time: float, duration: float, fps: float) -> Optional[List[np.ndarray]]:
        """ViViT용 프레임 추출 (32프레임)"""
        try:
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            total_clip_frames = end_frame - start_frame

            if total_clip_frames < 32:
                logger.warning(f"클립 프레임 수가 부족합니다: {total_clip_frames}")
                return None

            # 32프레임 균등 샘플링을 위한 인덱스 계산
            frame_indices = np.linspace(start_frame, end_frame - 1, 32, dtype=int)

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"프레임 {frame_idx} 읽기 실패")
                    continue

                # BGR to RGB 변환 및 크기 조정
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))

                # 0-255 범위를 0-1로 정규화
                frame_normalized = frame_resized.astype(np.float32) / 255.0

                frames.append(frame_normalized)

            if len(frames) != 32:
                logger.warning(f"추출된 프레임 수가 32개가 아닙니다: {len(frames)}")
                return None

            return frames

        except Exception as e:
            logger.error(f"프레임 추출 실패: {e}")
            return None

    def _extract_original_frames_for_saving(self, cap, start_time: float, duration: float, fps: float) -> Optional[List[np.ndarray]]:
        """원본 해상도 프레임 추출 (저장용, 32프레임)"""
        try:
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            total_clip_frames = end_frame - start_frame

            if total_clip_frames < 32:
                logger.warning(f"클립 프레임 수가 부족합니다: {total_clip_frames}")
                return None

            # 32프레임 균등 샘플링을 위한 인덱스 계산
            frame_indices = np.linspace(start_frame, end_frame - 1, 32, dtype=int)

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"프레임 {frame_idx} 읽기 실패")
                    continue

                # BGR to RGB 변환만 (리사이징과 정규화 없음)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            if len(frames) != 32:
                logger.warning(f"추출된 프레임 수가 32개가 아닙니다: {len(frames)}")
                return None

            return frames

        except Exception as e:
            logger.error(f"원본 프레임 추출 실패: {e}")
            return None

    def predict(self, video_bytes: bytes, audit_id: str = None, save_clips: bool = False) -> Dict[str, Any]:
        """비디오 진단 수행"""
        if not self.model_loaded:
            raise RuntimeError("모델이 로딩되지 않았습니다")

        if self.processor is None:
            raise RuntimeError("프로세서가 로딩되지 않았습니다")

        try:
            # 비디오 전처리
            clips_list = self.preprocess_video(video_bytes)

            # 클립 저장 (옵션)
            saved_clip_paths = []
            if save_clips and audit_id:
                saved_clip_paths = self.save_clips_as_videos(clips_list, audit_id)

            all_predictions = []
            all_confidences = []

            # 각 클립에 대해 예측 수행
            for i, clip_frames in enumerate(clips_list):
                try:
                    logger.debug(f"클립 {i+1} 처리 시작 - 프레임 수: {len(clip_frames)}")

                    # ViViT 프로세서로 전처리
                    try:
                        inputs = self.processor(
                            images=clip_frames,  # images 파라미터 사용
                            return_tensors="pt"
                        )
                        logger.debug(f"클립 {i+1}: 프로세서 처리 완료")
                    except Exception as proc_error:
                        logger.error(f"클립 {i+1}: 프로세서 오류 - {proc_error}")
                        # 대안: 비디오 프레임을 하나씩 처리
                        try:
                            # numpy 배열을 직접 전달
                            inputs = self.processor(
                                clip_frames,  # 위치 인수로 전달
                                return_tensors="pt"
                            )
                            logger.debug(f"클립 {i+1}: 대안 프로세서 처리 완료")
                        except Exception as alt_error:
                            logger.error(f"클립 {i+1}: 모든 프로세서 방법 실패 - {alt_error}")
                            continue

                    # 디바이스로 이동
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    logger.debug(f"클립 {i+1}: 디바이스 이동 완료")

                    # 예측 수행
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        probabilities = torch.nn.functional.softmax(logits, dim=-1)

                        predicted_class = torch.argmax(probabilities, dim=-1).item()
                        confidence = torch.max(probabilities).item()

                        all_predictions.append(predicted_class)
                        all_confidences.append(confidence)

                        logger.info(f"클립 {i+1}: {self.class_mapping[predicted_class]} (신뢰도: {confidence:.3f})")

                except Exception as e:
                    logger.error(f"클립 {i+1} 처리 중 오류: {e}")
                    import traceback
                    logger.error(f"상세 오류: {traceback.format_exc()}")
                    continue

            if not all_predictions:
                raise RuntimeError("모든 클립 처리에 실패했습니다")

            # 클래스별 신뢰도 그룹화 및 평균 계산
            class_confidences = {}
            for pred, conf in zip(all_predictions, all_confidences):
                class_name = self.class_mapping[pred]
                if class_name not in class_confidences:
                    class_confidences[class_name] = []
                class_confidences[class_name].append(conf)

            # 각 클래스별 평균 신뢰도 계산
            class_avg_confidences = {}
            for class_name, confidences in class_confidences.items():
                avg_confidence = np.mean(confidences)
                class_avg_confidences[class_name] = {
                    'avg_confidence': avg_confidence,
                    'count': len(confidences),
                    'confidences': confidences
                }
                logger.info(f"클래스 '{class_name}': 평균 신뢰도 {avg_confidence:.3f} ({len(confidences)}개 클립)")

            # 가장 높은 평균 신뢰도를 가진 클래스 선택
            best_class = max(class_avg_confidences.keys(),
                           key=lambda k: class_avg_confidences[k]['avg_confidence'])

            final_class_name = best_class
            final_confidence = class_avg_confidences[best_class]['avg_confidence']
            prediction_count = class_avg_confidences[best_class]['count']

            # 불량 여부 판단 (normal이 아닌 경우 불량)
            is_defect = final_class_name != 'normal'

            # 상세 진단 결과 생성
            diagnosis_details = {
                'predicted_class': final_class_name,
                'avg_confidence': final_confidence,
                'prediction_count': prediction_count,
                'total_clips': len(clips_list),
                'class_statistics': class_avg_confidences,
                'predictions_per_clip': [
                    {
                        'clip_index': i,
                        'predicted_class': self.class_mapping[pred],
                        'confidence': conf
                    }
                    for i, (pred, conf) in enumerate(zip(all_predictions, all_confidences))
                ]
            }

            diagnosis_result = f"와이퍼 상태: {final_class_name} (평균 신뢰도: {final_confidence:.3f}, {prediction_count}/{len(clips_list)} 클립)"
            if is_defect:
                diagnosis_result += f" - 불량 감지됨"
            else:
                diagnosis_result += f" - 정상 상태"

            result = {
                'is_defect': is_defect,
                'predicted_class': final_class_name,
                'confidence': final_confidence,
                'diagnosis_result': diagnosis_result,
                'details': diagnosis_details,
                'saved_clips': saved_clip_paths if save_clips else []
            }

            logger.info(f"최종 진단 결과: {diagnosis_result}")
            return result

        except Exception as e:
            logger.error(f"예측 실패: {e}")
            raise

    def save_clips_as_videos(self, clips_list: List[List[np.ndarray]], audit_id: str, output_dir: str = "output") -> List[str]:
        """샘플링된 클립들을 mp4 파일로 저장"""
        saved_paths = []

        try:
            # output 디렉토리 생성
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            for i, clip_frames in enumerate(clips_list):
                # 파일명 생성
                video_filename = f"{audit_id}_clip_{i+1:02d}.mp4"
                video_path = output_path / video_filename

                # VideoWriter 설정
                height, width = clip_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 16  # 32프레임을 2초로 재생 (32/2 = 16fps)

                out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

                if not out.isOpened():
                    logger.error(f"VideoWriter 초기화 실패: {video_path}")
                    continue

                # 프레임 쓰기
                for frame in clip_frames:
                    # 정규화된 프레임을 0-255 범위로 변환
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    # RGB를 BGR로 변환 (OpenCV 요구사항)
                    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()
                saved_paths.append(str(video_path))
                logger.info(f"클립 {i+1} 저장 완료: {video_path}")

            return saved_paths

        except Exception as e:
            logger.error(f"비디오 클립 저장 실패: {e}")
            return saved_paths

# 전역 모델 인스턴스
model_instance = WiperModel()

def get_model() -> WiperModel:
    """모델 인스턴스 반환"""
    return model_instance

def initialize_model() -> bool:
    """모델 초기화"""
    model_path = settings.MODEL_PATH
    return model_instance.load_model(model_path)
