"""
추론 후처리 파이프라인 모듈
다양한 후처리 방법론을 개별적으로 선택하거나 조합할 수 있는 파이프라인 제공
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from collections import Counter
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PostProcessorBase(ABC):
    """후처리기 베이스 클래스"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, predictions: List[Dict[str, Any]], total_clips: int) -> Dict[str, Any]:
        """예측 결과를 후처리"""
        pass

    def get_name(self) -> str:
        return self.name


class SimpleAggregator(PostProcessorBase):
    """단순 다수결 후처리기"""

    def __init__(self):
        super().__init__("simple_majority")

    def process(self, predictions: List[Dict[str, Any]], total_clips: int) -> Dict[str, Any]:
        """단순 다수결 처리"""
        class_votes = [pred['predicted_class'] for pred in predictions]
        vote_counts = Counter(class_votes)

        final_class = vote_counts.most_common(1)[0][0]
        vote_ratio = vote_counts[final_class] / len(predictions)

        class_confidences = [pred['confidence'] for pred in predictions if pred['predicted_class'] == final_class]
        avg_confidence = np.mean(class_confidences)

        is_defect = final_class != 'normal'

        return {
            'is_defect': is_defect,
            'predicted_class': final_class,
            'confidence': avg_confidence,
            'diagnosis_result': f"다수결 결과: {final_class} (득표율: {vote_ratio:.1%}, 평균 신뢰도: {avg_confidence:.3f})",
            'method_info': {
                'method': self.name,
                'vote_counts': dict(vote_counts),
                'vote_ratio': vote_ratio
            },
            'total_clips': total_clips,
            'successful_clips': len(predictions)
        }


class ConfidenceWeightedAggregator(PostProcessorBase):
    """신뢰도 가중 후처리기"""

    def __init__(self, weight_power: float = 2.0):
        super().__init__("confidence_weighted")
        self.weight_power = weight_power  # 신뢰도 가중치 제곱수

    def process(self, predictions: List[Dict[str, Any]], total_clips: int) -> Dict[str, Any]:
        """신뢰도 가중 평균 처리"""
        class_weighted_scores = {}

        for pred in predictions:
            class_name = pred['predicted_class']
            confidence = pred['confidence']

            if class_name not in class_weighted_scores:
                class_weighted_scores[class_name] = {'total_score': 0.0, 'count': 0, 'confidences': []}

            # 신뢰도의 제곱으로 가중치 적용
            weighted_score = confidence ** self.weight_power
            class_weighted_scores[class_name]['total_score'] += weighted_score
            class_weighted_scores[class_name]['count'] += 1
            class_weighted_scores[class_name]['confidences'].append(confidence)

        # 평균 가중 점수가 가장 높은 클래스 선택
        best_class = max(class_weighted_scores.keys(),
                        key=lambda k: class_weighted_scores[k]['total_score'] / class_weighted_scores[k]['count'])

        best_score = class_weighted_scores[best_class]['total_score'] / class_weighted_scores[best_class]['count']
        avg_confidence = np.mean(class_weighted_scores[best_class]['confidences'])

        is_defect = best_class != 'normal'

        return {
            'is_defect': is_defect,
            'predicted_class': best_class,
            'confidence': avg_confidence,
            'diagnosis_result': f"신뢰도 가중 결과: {best_class} (가중 점수: {best_score:.3f}, 평균 신뢰도: {avg_confidence:.3f})",
            'method_info': {
                'method': self.name,
                'class_scores': {k: v['total_score']/v['count'] for k, v in class_weighted_scores.items()},
                'weighted_score': best_score,
                'weight_power': self.weight_power
            },
            'total_clips': total_clips,
            'successful_clips': len(predictions)
        }


class ThresholdBasedFilter(PostProcessorBase):
    """임계값 기반 필터"""

    def __init__(self, confidence_threshold: float = 0.7, min_defect_ratio: float = 0.3):
        super().__init__("threshold_based")
        self.confidence_threshold = confidence_threshold
        self.min_defect_ratio = min_defect_ratio

    def process(self, predictions: List[Dict[str, Any]], total_clips: int) -> Dict[str, Any]:
        """임계값 기반 필터링"""
        # 높은 신뢰도의 결함 예측만 필터링
        high_confidence_defects = []
        for pred in predictions:
            if pred['predicted_class'] != 'normal' and pred['confidence'] >= self.confidence_threshold:
                high_confidence_defects.append(pred)

        defect_ratio = len(high_confidence_defects) / len(predictions)

        if defect_ratio >= self.min_defect_ratio and high_confidence_defects:
            # 가장 높은 신뢰도의 결함 선택
            best_defect = max(high_confidence_defects, key=lambda x: x['confidence'])
            final_class = best_defect['predicted_class']
            final_confidence = best_defect['confidence']
            is_defect = True
        else:
            # 정상으로 판정
            normal_predictions = [pred for pred in predictions if pred['predicted_class'] == 'normal']
            if normal_predictions:
                final_confidence = np.mean([pred['confidence'] for pred in normal_predictions])
            else:
                final_confidence = 0.5
            final_class = 'normal'
            is_defect = False

        return {
            'is_defect': is_defect,
            'predicted_class': final_class,
            'confidence': final_confidence,
            'diagnosis_result': f"임계값 기반 결과: {final_class} (결함 비율: {defect_ratio:.1%}, 신뢰도: {final_confidence:.3f})",
            'method_info': {
                'method': self.name,
                'confidence_threshold': self.confidence_threshold,
                'defect_ratio': defect_ratio,
                'min_defect_ratio': self.min_defect_ratio,
                'high_confidence_defects': len(high_confidence_defects)
            },
            'total_clips': total_clips,
            'successful_clips': len(predictions)
        }


class TemporalConsistencyAnalyzer(PostProcessorBase):
    """시간적 일관성 분석기"""

    def __init__(self, min_sequence_length: int = 3, coverage_weight: float = 0.5):
        super().__init__("temporal_consistency")
        self.min_sequence_length = min_sequence_length
        self.coverage_weight = coverage_weight  # 커버리지 가중치

    def process(self, predictions: List[Dict[str, Any]], total_clips: int) -> Dict[str, Any]:
        """시간적 일관성 분석"""
        class_sequences = {}

        for class_name in set(pred['predicted_class'] for pred in predictions):
            # 해당 클래스가 예측된 클립 인덱스들
            class_clips = [pred['clip_index'] for pred in predictions if pred['predicted_class'] == class_name]
            class_clips.sort()

            # 연속 시퀀스 길이 계산
            sequences = self._find_sequences(class_clips)

            if sequences:
                # 가장 긴 연속 시퀀스
                longest_seq = max(sequences, key=len)
                avg_confidence = np.mean([pred['confidence'] for pred in predictions
                                        if pred['clip_index'] in longest_seq and pred['predicted_class'] == class_name])

                # 시간적 일관성 점수 계산 (길이 + 커버리지)
                consistency_score = len(longest_seq) + (len(longest_seq) / total_clips) * self.coverage_weight

                class_sequences[class_name] = {
                    'max_sequence_length': len(longest_seq),
                    'sequence_count': len(sequences),
                    'avg_confidence': avg_confidence,
                    'coverage': len(longest_seq) / total_clips,
                    'consistency_score': consistency_score
                }

        # 일관성 점수가 가장 높은 클래스 선택
        if class_sequences:
            best_class = max(class_sequences.keys(),
                           key=lambda k: class_sequences[k]['consistency_score'])
            best_info = class_sequences[best_class]

            final_class = best_class
            final_confidence = best_info['avg_confidence']
            is_defect = final_class != 'normal'
        else:
            # 연속 시퀀스가 없으면 단순 다수결로 대체
            fallback = SimpleAggregator()
            return fallback.process(predictions, total_clips)

        return {
            'is_defect': is_defect,
            'predicted_class': final_class,
            'confidence': final_confidence,
            'diagnosis_result': f"시간적 일관성 결과: {final_class} (최대 연속길이: {best_info['max_sequence_length']}, 커버리지: {best_info['coverage']:.1%})",
            'method_info': {
                'method': self.name,
                'class_sequences': class_sequences,
                'min_sequence_length': self.min_sequence_length,
                'coverage_weight': self.coverage_weight
            },
            'total_clips': total_clips,
            'successful_clips': len(predictions)
        }

    def _find_sequences(self, clip_indices: List[int]) -> List[List[int]]:
        """연속 시퀀스 찾기"""
        if not clip_indices:
            return []

        sequences = []
        current_seq = [clip_indices[0]]

        for i in range(1, len(clip_indices)):
            if clip_indices[i] - clip_indices[i-1] == 1:  # 연속
                current_seq.append(clip_indices[i])
            else:
                if len(current_seq) >= self.min_sequence_length:
                    sequences.append(current_seq)
                current_seq = [clip_indices[i]]

        if len(current_seq) >= self.min_sequence_length:
            sequences.append(current_seq)

        return sequences


class InferencePipeline:
    """추론 후처리 파이프라인"""

    def __init__(self):
        self.processors: List[PostProcessorBase] = []
        self.ensemble_mode = False
        self.ensemble_weights: Dict[str, float] = {}

    def add_processor(self, processor: PostProcessorBase, weight: float = 1.0):
        """후처리기 추가"""
        self.processors.append(processor)
        self.ensemble_weights[processor.get_name()] = weight
        logger.info(f"후처리기 추가: {processor.get_name()} (가중치: {weight})")

    def set_ensemble_mode(self, enabled: bool):
        """앙상블 모드 설정"""
        self.ensemble_mode = enabled
        logger.info(f"앙상블 모드: {'활성화' if enabled else '비활성화'}")

    def process(self, predictions: List[Dict[str, Any]], total_clips: int) -> Dict[str, Any]:
        """파이프라인 실행"""
        if not self.processors:
            raise ValueError("후처리기가 등록되지 않았습니다")

        if not self.ensemble_mode:
            # 단일 후처리기 사용 (첫 번째)
            processor = self.processors[0]
            result = processor.process(predictions, total_clips)
            logger.info(f"단일 후처리 완료: {processor.get_name()}")
            return result
        else:
            # 앙상블 모드
            return self._ensemble_process(predictions, total_clips)

    def _ensemble_process(self, predictions: List[Dict[str, Any]], total_clips: int) -> Dict[str, Any]:
        """앙상블 처리"""
        results = []
        processor_names = []

        # 각 후처리기 실행
        for processor in self.processors:
            result = processor.process(predictions, total_clips)
            results.append(result)
            processor_names.append(processor.get_name())

        # 가중 투표
        class_scores = {}

        for i, result in enumerate(results):
            processor_name = processor_names[i]
            weight = self.ensemble_weights.get(processor_name, 1.0)
            predicted_class = result['predicted_class']
            confidence = result['confidence']

            if predicted_class not in class_scores:
                class_scores[predicted_class] = {'total_score': 0.0, 'count': 0, 'confidences': []}

            # 가중 점수 = 신뢰도 × 후처리기 가중치
            weighted_score = confidence * weight
            class_scores[predicted_class]['total_score'] += weighted_score
            class_scores[predicted_class]['count'] += 1
            class_scores[predicted_class]['confidences'].append(confidence)

        # 최고 점수 클래스 선택
        best_class = max(class_scores.keys(),
                        key=lambda k: class_scores[k]['total_score'])

        final_confidence = np.mean(class_scores[best_class]['confidences'])
        is_defect = best_class != 'normal'

        # 개별 결과 정보
        individual_results = {name: result['predicted_class'] for name, result in zip(processor_names, results)}

        return {
            'is_defect': is_defect,
            'predicted_class': best_class,
            'confidence': final_confidence,
            'diagnosis_result': f"앙상블 결과: {best_class} (참여 후처리기: {len(self.processors)}개, 신뢰도: {final_confidence:.3f})",
            'method_info': {
                'method': 'ensemble',
                'individual_results': individual_results,
                'class_scores': {k: v['total_score'] for k, v in class_scores.items()},
                'processor_weights': self.ensemble_weights.copy(),
                'participating_processors': processor_names
            },
            'total_clips': total_clips,
            'successful_clips': len(predictions)
        }


class PipelineBuilder:
    """파이프라인 빌더 - 간편한 설정을 위한 헬퍼 클래스"""

    @staticmethod
    def create_simple_pipeline() -> InferencePipeline:
        """단순 다수결 파이프라인"""
        pipeline = InferencePipeline()
        pipeline.add_processor(SimpleAggregator())
        return pipeline

    @staticmethod
    def create_confidence_pipeline(weight_power: float = 2.0) -> InferencePipeline:
        """신뢰도 가중 파이프라인"""
        pipeline = InferencePipeline()
        pipeline.add_processor(ConfidenceWeightedAggregator(weight_power=weight_power))
        return pipeline

    @staticmethod
    def create_threshold_pipeline(confidence_threshold: float = 0.7,
                                min_defect_ratio: float = 0.3) -> InferencePipeline:
        """임계값 기반 파이프라인"""
        pipeline = InferencePipeline()
        pipeline.add_processor(ThresholdBasedFilter(confidence_threshold, min_defect_ratio))
        return pipeline

    @staticmethod
    def create_temporal_pipeline(min_sequence_length: int = 3,
                               coverage_weight: float = 0.5) -> InferencePipeline:
        """시간적 일관성 파이프라인"""
        pipeline = InferencePipeline()
        pipeline.add_processor(TemporalConsistencyAnalyzer(min_sequence_length, coverage_weight))
        return pipeline

    @staticmethod
    def create_ensemble_pipeline(confidence_threshold: float = 0.7,
                               min_defect_ratio: float = 0.3,
                               min_sequence_length: int = 3,
                               processor_weights: Optional[Dict[str, float]] = None) -> InferencePipeline:
        """앙상블 파이프라인"""
        pipeline = InferencePipeline()

        # 기본 가중치
        default_weights = {
            'simple_majority': 1.0,
            'confidence_weighted': 1.2,
            'threshold_based': 1.5,
            'temporal_consistency': 1.3
        }

        weights = processor_weights or default_weights

        # 후처리기들 추가
        pipeline.add_processor(SimpleAggregator(), weights.get('simple_majority', 1.0))
        pipeline.add_processor(ConfidenceWeightedAggregator(), weights.get('confidence_weighted', 1.0))
        pipeline.add_processor(ThresholdBasedFilter(confidence_threshold, min_defect_ratio),
                             weights.get('threshold_based', 1.0))
        pipeline.add_processor(TemporalConsistencyAnalyzer(min_sequence_length),
                             weights.get('temporal_consistency', 1.0))

        pipeline.set_ensemble_mode(True)
        return pipeline

    @staticmethod
    def create_custom_pipeline(processors_config: List[Dict[str, Any]]) -> InferencePipeline:
        """커스텀 파이프라인 생성

        Args:
            processors_config: 후처리기 설정 목록
                예: [
                    {'type': 'confidence_weighted', 'weight': 1.2, 'params': {'weight_power': 2.0}},
                    {'type': 'threshold_based', 'weight': 1.5, 'params': {'confidence_threshold': 0.8}}
                ]
        """
        pipeline = InferencePipeline()

        for config in processors_config:
            processor_type = config['type']
            weight = config.get('weight', 1.0)
            params = config.get('params', {})

            if processor_type == 'simple_majority':
                processor = SimpleAggregator()
            elif processor_type == 'confidence_weighted':
                processor = ConfidenceWeightedAggregator(**params)
            elif processor_type == 'threshold_based':
                processor = ThresholdBasedFilter(**params)
            elif processor_type == 'temporal_consistency':
                processor = TemporalConsistencyAnalyzer(**params)
            else:
                raise ValueError(f"지원하지 않는 후처리기 타입: {processor_type}")

            pipeline.add_processor(processor, weight)

        # 2개 이상이면 앙상블 모드 활성화
        if len(processors_config) > 1:
            pipeline.set_ensemble_mode(True)

        return pipeline
