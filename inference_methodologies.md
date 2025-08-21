# 영상 기반 결함 감지 - 추론 방법론 가이드

## 현재 구현된 5가지 방법론

### 1. Simple Majority Voting (단순 다수결) - `simple_majority`
```python
# 가장 많이 예측된 클래스를 선택
AGGREGATION_METHOD = "simple_majority"
```
**장점**: 구현 단순, 해석 용이, 빠른 처리
**단점**: 신뢰도 무시, 노이즈에 취약
**적용상황**: 데이터가 깨끗하고 결함이 전체 영상에 걸쳐 나타날 때
**출력 예시**: `다수결 결과: left_fail (득표율: 60.0%, 평균 신뢰도: 0.850)`

### 2. Confidence Weighted Averaging (신뢰도 가중 평균) - `confidence_weighted`
```python
# 각 클래스별로 신뢰도 제곱 가중 평균 계산
AGGREGATION_METHOD = "confidence_weighted"
```
**장점**: 신뢰도 높은 예측에 더 큰 가중치, 모델 불확실성 반영
**단점**: 이상치에 민감할 수 있음
**적용상황**: 모델의 신뢰도가 신뢰할 만하고 품질이 일정할 때
**출력 예시**: `신뢰도 가중 결과: right_fail (가중 점수: 0.723, 평균 신뢰도: 0.850)`

### 3. Threshold-Based Detection (임계값 기반) - `threshold_based`
```python
# 설정 가능한 파라미터
AGGREGATION_METHOD = "threshold_based"
CONFIDENCE_THRESHOLD = 0.7  # 이 값 이상의 신뢰도만 인정
MIN_DEFECT_RATIO = 0.3      # 전체 클립 중 이 비율 이상에서 결함 감지되어야 함
```
**장점**: False Positive 줄임, 확실한 결함만 감지, 보수적 판정
**단점**: 임계값 설정에 민감, False Negative 증가 가능
**적용상황**: 정밀도가 중요하고, 확실하지 않은 결함은 놓쳐도 되는 경우
**출력 예시**: `임계값 기반 결과: wiper_stop (결함 비율: 40.0%, 신뢰도: 0.920)`

### 4. Temporal Consistency (시간적 일관성) - `temporal_consistency`
```python
# 설정 가능한 파라미터
AGGREGATION_METHOD = "temporal_consistency"
MIN_SEQUENCE_LENGTH = 3  # 최소 이 길이 이상 연속으로 나타나야 인정
```
**장점**: 노이즈 필터링, 실제 지속적 결함 패턴 감지, 일시적 오류 제거
**단점**: 순간적 결함 놓칠 수 있음, 짧은 영상에서 효과 제한적
**적용상황**: 결함이 연속적으로 나타나는 특성이 있고, 노이즈가 많은 환경
**출력 예시**: `시간적 일관성 결과: angle_limit (최대 연속길이: 5, 커버리지: 50.0%)`

### 5. Ensemble Methods (앙상블) - `ensemble`
```python
# 3가지 방법(다수결, 신뢰도가중, 임계값)을 조합하여 투표
AGGREGATION_METHOD = "ensemble"
```
**장점**: 강건한 예측, 각 방법의 장점 활용, 높은 정확도
**단점**: 복잡성 증가, 해석 어려움, 처리 시간 증가
**적용상황**: 높은 정확도가 필요하고 복잡성을 감당할 수 있을 때, 프로덕션 환경
**출력 예시**: `앙상블 결과: slow (결함 투표: 2/3, 신뢰도: 0.780)`

## 설정 방법

### 1. 환경변수로 설정 (.env 파일)
```bash
# 기본 방법 선택
AGGREGATION_METHOD=simple_majority

# 세부 파라미터 (해당 방법 사용 시만 적용)
CONFIDENCE_THRESHOLD=0.8
MIN_DEFECT_RATIO=0.2
MIN_SEQUENCE_LENGTH=4
```

### 2. config.py에서 직접 설정
```python
class Settings(BaseSettings):
    AGGREGATION_METHOD: str = "temporal_consistency"
    CONFIDENCE_THRESHOLD: float = 0.75
    MIN_DEFECT_RATIO: float = 0.25
    MIN_SEQUENCE_LENGTH: int = 3
```

## 방법론별 상세 결과 정보

각 방법론은 공통 필드 외에 고유한 분석 정보를 제공합니다:

### Simple Majority
```json
{
  "method_info": {
    "method": "simple_majority",
    "vote_counts": {"normal": 4, "left_fail": 6},
    "vote_ratio": 0.6
  }
}
```

### Confidence Weighted
```json
{
  "method_info": {
    "method": "confidence_weighted", 
    "class_scores": {"normal": 0.65, "left_fail": 0.78},
    "weighted_score": 0.78
  }
}
```

### Threshold Based
```json
{
  "method_info": {
    "method": "threshold_based",
    "confidence_threshold": 0.7,
    "defect_ratio": 0.4,
    "min_defect_ratio": 0.3,
    "high_confidence_defects": 4
  }
}
```

### Temporal Consistency
```json
{
  "method_info": {
    "method": "temporal_consistency",
    "class_sequences": {
      "left_fail": {
        "max_sequence_length": 5,
        "sequence_count": 2,
        "avg_confidence": 0.85,
        "coverage": 0.5
      }
    },
    "min_sequence_length": 3
  }
}
```

### Ensemble
```json
{
  "method_info": {
    "method": "ensemble",
    "individual_results": {
      "majority": "left_fail",
      "confidence_weighted": "left_fail", 
      "threshold_based": "normal"
    },
    "defect_votes": 2
  }
}
```

## 실제 사용 시나리오별 추천

### 🏭 **생산라인 실시간 검사** 
→ `simple_majority` 또는 `confidence_weighted`
- 빠른 처리 속도 필요
- 안정된 환경에서 노이즈 적음

### 🔬 **품질관리 정밀검사**
→ `threshold_based` 또는 `ensemble`  
- 정확도 최우선
- False Positive 최소화 중요

### 📹 **현장 모니터링 (노이즈 많음)**
→ `temporal_consistency` 또는 `ensemble`
- 일시적 오분류 많음
- 지속적 패턴만 감지하고 싶음

### ⚡ **실시간 알람 시스템**
→ `threshold_based` (높은 임계값)
- 확실한 결함만 알람
- False Alarm 방지 중요

### 🧪 **연구/개발 단계**
→ `ensemble` + 상세 로그 분석
- 다양한 방법 비교 분석
- 최적 방법론 선택을 위한 실험

## 튜닝 가이드

### Threshold-Based 튜닝
```python
# 보수적 (정밀도 우선)
CONFIDENCE_THRESHOLD = 0.8
MIN_DEFECT_RATIO = 0.4

# 민감한 (재현율 우선)  
CONFIDENCE_THRESHOLD = 0.6
MIN_DEFECT_RATIO = 0.2
```

### Temporal Consistency 튜닝
```python
# 짧은 영상 (5-10초)
MIN_SEQUENCE_LENGTH = 2

# 긴 영상 (30초+)
MIN_SEQUENCE_LENGTH = 5
```

이제 원하는 방법론을 선택하여 적용할 수 있으며, 각 방법론의 특성을 이해하고 상황에 맞게 튜닝할 수 있습니다.
