import torch
import logging
from pathlib import Path
from typing import Optional
from transformers import VivitImageProcessor, VivitForVideoClassification

logger = logging.getLogger(__name__)

class ModelLoader:
    """ViViT 모델 로딩을 담당하는 클래스"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ModelLoader 초기화 - 디바이스: {self.device}")

    def load_processor(self) -> VivitImageProcessor:
        """ViViT 프로세서 로딩"""
        try:
            logger.info("ViViT 프로세서 로딩 시작...")
            processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
            logger.info("ViViT 프로세서 로딩 완료")
            return processor
        except Exception as e:
            logger.error(f"프로세서 로딩 실패: {e}")
            raise

    def load_base_model(self, num_labels: int = 7) -> VivitForVideoClassification:
        """기본 ViViT 모델 로딩"""
        try:
            logger.info("ViViT 기본 모델 로딩 시작...")
            model = VivitForVideoClassification.from_pretrained(
                "google/vivit-b-16x2-kinetics400",
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            logger.info("ViViT 기본 모델 로딩 완료")
            return model
        except Exception as e:
            logger.error(f"기본 모델 로딩 실패: {e}")
            raise

    def load_checkpoint(self, model: VivitForVideoClassification, model_path: str) -> VivitForVideoClassification:
        """파인튜닝된 가중치 로딩"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

            logger.info(f"파인튜닝된 가중치 로딩 시작: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # 가중치 적용
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    logger.warning(f"누락된 키: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"예상치 못한 키: {unexpected_keys}")

                logger.info("파인튜닝된 가중치 적용 완료")
            else:
                # 완전한 모델 객체인 경우
                model = checkpoint
                logger.info("완전한 모델 객체 로딩 완료")

            return model

        except Exception as e:
            logger.error(f"가중치 로딩 실패: {e}")
            raise

    def setup_model_for_inference(self, model: VivitForVideoClassification) -> VivitForVideoClassification:
        """모델을 추론 모드로 설정"""
        try:
            model.eval()
            model.to(self.device)
            logger.info(f"모델을 {self.device}로 이동하고 평가 모드로 설정 완료")
            logger.info(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
            return model
        except Exception as e:
            logger.error(f"모델 설정 실패: {e}")
            raise

    def load_complete_model(self, model_path: str, num_labels: int = 7) -> tuple[VivitForVideoClassification, VivitImageProcessor]:
        """완전한 모델 로딩 파이프라인"""
        try:
            # 프로세서 로딩
            processor = self.load_processor()

            # 기본 모델 로딩
            model = self.load_base_model(num_labels)

            # 체크포인트 로딩
            model = self.load_checkpoint(model, model_path)

            # 추론 설정
            model = self.setup_model_for_inference(model)

            logger.info("완전한 모델 로딩 파이프라인 완료")
            return model, processor

        except Exception as e:
            logger.error(f"완전한 모델 로딩 실패: {e}")
            raise
