import torch
import numpy as np
import logging
import tempfile
import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
import cv2

logger = logging.getLogger(__name__)

class WiperAI:
    """와이퍼 AI - 단순화된 버전"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.model_loaded = False

        # 클래스 매핑
        self.class_mapping = {
            0: 'angle_limit',
            1: 'left_fail',
            2: 'normal',
            3: 'right_wiper_fail',
            4: 'slow',
            5: 'wiper_lag',
            6: 'wiper_stop'
        }

        # 설정
        self.clip_duration = 2.0
        self.stride = 1.0
        self.num_frames = 32

    def load_model(self) -> bool:
        """모델 로딩"""
        try:
            logger.info(f"모델 로딩 시작: {self.model_path}")

            # 1. 프로세서 로딩
            self.processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

            # 2. 모델 구조 생성
            config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400", num_labels=7)
            self.model = VivitForVideoClassification(config)

            # 3. 파인튜닝된 가중치 로딩
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint

            self.model.load_state_dict(state_dict, strict=False)

            # 4. 추론 모드 설정
            self.model.eval()
            self.model.to(self.device)

            self.model_loaded = True
            logger.info(f"모델 로딩 완료 - 디바이스: {self.device}")
            return True

        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            return False

    def predict(self, video_bytes: bytes) -> Dict[str, Any]:
        """비디오 진단 수행"""
        if not self.model_loaded:
            raise RuntimeError("모델이 로딩되지 않았습니다")

        try:
            start_time = time.time()

            # 1. 비디오 전처리
            clips = self._extract_clips(video_bytes)

            # 2. 각 클립 추론
            predictions = []
            for i, clip_frames in enumerate(clips):
                result = self._predict_clip(clip_frames, i + 1)
                if result:
                    predictions.append(result)

            if not predictions:
                raise RuntimeError("모든 클립 처리에 실패했습니다")

            # 3. 결과 통합
            final_result = self._aggregate_results(predictions, len(clips))
            final_result['processing_time'] = time.time() - start_time

            return final_result

        except Exception as e:
            logger.error(f"예측 실패: {e}")
            raise

    def _extract_clips(self, video_bytes: bytes) -> List[List[np.ndarray]]:
        """비디오에서 클립 추출"""
        tmp_file_path = None
        try:
            # 임시 파일 생성
            tmp_fd, tmp_file_path = tempfile.mkstemp(suffix='.mp4')
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

            # 클립 추출
            clips = []
            start_time = 0.0

            while start_time + self.clip_duration <= duration:
                clip_frames = self._extract_frames(cap, start_time, fps)
                if clip_frames:
                    clips.append(clip_frames)
                start_time += self.stride

            cap.release()
            cv2.destroyAllWindows()

            logger.info(f"추출된 클립 수: {len(clips)}")
            return clips

        except Exception as e:
            logger.error(f"클립 추출 실패: {e}")
            raise
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    time.sleep(0.1)
                    os.unlink(tmp_file_path)
                except:
                    pass

    def _extract_frames(self, cap, start_time: float, fps: float) -> Optional[List[np.ndarray]]:
        """프레임 추출"""
        try:
            start_frame = int(start_time * fps)
            end_frame = int((start_time + self.clip_duration) * fps)
            total_frames = end_frame - start_frame

            if total_frames < self.num_frames:
                return None

            # 균등 간격으로 프레임 선택
            step = total_frames / self.num_frames
            frame_indices = [start_frame + int(i * step) for i in range(self.num_frames)]

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            return frames if len(frames) == self.num_frames else None

        except Exception as e:
            logger.error(f"프레임 추출 실패: {e}")
            return None

    def _predict_clip(self, clip_frames: List[np.ndarray], clip_index: int) -> Optional[Dict[str, Any]]:
        """단일 클립 추론"""
        try:
            # 전처리
            inputs = self.processor(images=clip_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()

                class_name = self.class_mapping[predicted_class]
                logger.info(f"클립 {clip_index}: {class_name} (신뢰도: {confidence:.3f})")

                return {
                    'predicted_class': class_name,
                    'confidence': confidence,
                    'clip_index': clip_index
                }

        except Exception as e:
            logger.error(f"클립 {clip_index} 추론 실패: {e}")
            return None

    def _aggregate_results(self, predictions: List[Dict[str, Any]], total_clips: int) -> Dict[str, Any]:
        """결과 통합"""
        # 클래스별 신뢰도 그룹화
        class_confidences = {}
        for pred in predictions:
            class_name = pred['predicted_class']
            if class_name not in class_confidences:
                class_confidences[class_name] = []
            class_confidences[class_name].append(pred['confidence'])

        # 최고 평균 신뢰도 클래스 선택
        best_class = max(class_confidences.keys(),
                        key=lambda k: np.mean(class_confidences[k]))

        final_confidence = np.mean(class_confidences[best_class])
        is_defect = best_class != 'normal'

        # 진단 결과 생성
        diagnosis = f"와이퍼 상태: {best_class} (신뢰도: {final_confidence:.3f})"
        if is_defect:
            diagnosis += " - 불량 감지됨"
        else:
            diagnosis += " - 정상 상태"

        return {
            'is_defect': is_defect,
            'predicted_class': best_class,
            'confidence': final_confidence,
            'diagnosis_result': diagnosis,
            'total_clips': total_clips,
            'successful_clips': len(predictions)
        }
