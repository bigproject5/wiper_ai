"""
Events module - 이벤트 처리 핸들러와 라우터
"""

import logging

# 패키지 초기화
logger = logging.getLogger(__name__)
logger.info("Events 패키지 초기화됨")

# 핵심 클래스들을 패키지 레벨에서 바로 import 가능하게 함
from .event_handler import BaseEventHandler
from .test_started_handler import TestStartedEventHandler

# 패키지에서 공개할 API 정의
__all__ = [
    'BaseEventHandler',
    'TestStartedEventHandler'
]
