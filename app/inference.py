import torch
import numpy as np
import logging
from typing import Dict, Any, Optional
from transformers import VivitImageProcessor, VivitForVideoClassification

from app.config import settings
from app.model_loader import ModelLoader
from app.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

class WiperModel:
    """ViViT 기반 와이퍼 진단 모델"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[VivitForVideoClassification] = None
        self.processor: Optional[VivitImageProcessor] = None
        self.model_loaded = False

        # 모듈 초기화
        self.model_loader = ModelLoader(self.device)
        self.video_processor = VideoProcessor(clip_duration=2.0, stride=1.0, num_frames=32)

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

        logger.info(f"WiperModel 초기화 완료 - 디바이스: {self.device}")

    def load_model(self, model_path: str) -> bool:
        """ViViT 모델 로딩"""
        try:
            logger.info(f"ViViT 모델 로딩 시작: {model_path}")

            # 완전한 모델 로딩 파이프라인 실행
            self.model, self.processor = self.model_loader.load_complete_model(model_path)

            # 최종 검증
            if self.processor is None or self.model is None:
                logger.error("모델 또는 프로세서 로딩 실패")
                return False

            self.model_loaded = True
            logger.info(f"ViViT 모델 로딩 완료")
            logger.info(f"프로세서 상태: {type(self.processor).__name__}")
            logger.info(f"모델 상태: {type(self.model).__name__}")

            return True

        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            self.model_loaded = False
            self.processor = None
            self.model = None
            return False

    def predict(self, video_bytes: bytes, audit_id: str = None, save_clips: bool = False) -> Dict[str, Any]:
        """비디오 진단 수행"""
        if not self.model_loaded:
            raise RuntimeError("모델이 로딩되지 않았습니다")

        if self.processor is None or self.model is None:
            raise RuntimeError("프로세서 또는 모델이 로딩되지 않았습니다")

        try:
            # 비디오 전처리
            clips_list = self.video_processor.preprocess_video(video_bytes)

            # 클립 저장 (옵션)
            saved_clip_paths = []
            if save_clips and audit_id:
                saved_clip_paths = self.video_processor.save_clips_as_videos(clips_list, audit_id)

            all_predictions = []
            all_confidences = []

            # 각 클립에 대해 예측 수행
            for i, clip_frames in enumerate(clips_list):
                try:
                    logger.debug(f"클립 {i+1} 처리 시작 - 프레임 수: {len(clip_frames)}")

                    # 학습 데이터 형태에 맞게 전처리
                    inputs = self._prepare_input_tensor(clip_frames, i+1)
                    if inputs is None:
                        continue

                    # 디바이스로 이동
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    logger.debug(f"클립 {i+1}: 디바이스 이동 완료")

                    # 예측 수행
                    prediction_result = self._perform_inference(inputs, i+1)
                    if prediction_result is not None:
                        predicted_class, confidence = prediction_result
                        all_predictions.append(predicted_class)
                        all_confidences.append(confidence)

                except Exception as e:
                    logger.error(f"클립 {i+1} 처리 중 오류: {e}")
                    import traceback
                    logger.error(f"상세 오류: {traceback.format_exc()}")
                    continue

            if not all_predictions:
                raise RuntimeError("모든 클립 처리에 실패했습니다")

            # 결과 분석 및 최종 진단
            result = self._analyze_predictions(all_predictions, all_confidences, clips_list, saved_clip_paths)
            return result

        except Exception as e:
            logger.error(f"예측 실패: {e}")
            raise

    def _prepare_input_tensor(self, clip_frames, clip_index: int) -> Optional[Dict[str, torch.Tensor]]:
        """클립 프레임을 모델 입력 텐서로 변환"""
        try:
            # 클립 프레임들을 올바른 형태로 변환
            clip_array = np.array(clip_frames)  # (32, 224, 224, 3)

            # 채널을 앞으로 이동: (32, 3, 224, 224)
            clip_array = np.transpose(clip_array, (0, 3, 1, 2))

            # 배치 차원 추가: (1, 32, 3, 224, 224) - 학습 데이터와 동일한 형태
            clip_batch = np.expand_dims(clip_array, axis=0)

            # torch 텐서로 변환
            clip_tensor = torch.from_numpy(clip_batch).float()

            # 직접 입력 딕셔너리 생성
            inputs = {'pixel_values': clip_tensor}

            logger.debug(f"클립 {clip_index}: 직접 처리 완료 - 입력 형태: {inputs['pixel_values'].shape}")
            return inputs

        except Exception as direct_error:
            logger.error(f"클립 {clip_index}: 직접 처리 오류 - {direct_error}")
            # 대안: ViViT 프로세서 사용
            try:
                inputs = self.processor(
                    images=clip_frames,
                    return_tensors="pt"
                )
                logger.debug(f"클립 {clip_index}: 프로세서 처리 완료 - 입력 형태: {inputs['pixel_values'].shape}")
                return inputs
            except Exception as alt_error:
                logger.error(f"클립 {clip_index}: 모든 처리 방법 실패 - {alt_error}")
                return None

    def _perform_inference(self, inputs: Dict[str, torch.Tensor], clip_index: int) -> Optional[tuple[int, float]]:
        """모델 추론 수행"""
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                # 모든 클래스의 확률 로깅
                class_probs = probabilities[0].cpu().numpy()

                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()

                # 상세 확률 정보 로깅
                prob_info = ", ".join([f"{self.class_mapping[j]}: {class_probs[j]:.3f}"
                                     for j in range(len(self.class_mapping))])
                logger.info(f"클립 {clip_index}: {self.class_mapping[predicted_class]} (신뢰도: {confidence:.3f})")
                logger.debug(f"클립 {clip_index} 전체 확률: {prob_info}")

                return predicted_class, confidence

        except Exception as e:
            logger.error(f"클립 {clip_index} 추론 실패: {e}")
            return None

    def _analyze_predictions(self, all_predictions, all_confidences, clips_list, saved_clip_paths) -> Dict[str, Any]:
        """예측 결과 분석 및 최종 진단 생성"""
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
            'saved_clips': saved_clip_paths
        }

        logger.info(f"최종 진단 결과: {diagnosis_result}")
        return result

# 전역 모델 인스턴스
model_instance = WiperModel()

def get_model() -> WiperModel:
    """모델 인스턴스 반환"""
    return model_instance

def initialize_model() -> bool:
    """모델 초기화"""
    model_path = settings.MODEL_PATH
    return model_instance.load_model(model_path)
