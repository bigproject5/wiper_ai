from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseEventHandler(ABC):
    """이벤트 핸들러 베이스 클래스"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def handle(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """이벤트 처리"""
        pass

    @abstractmethod
    def get_event_type(self) -> str:
        """처리할 이벤트 타입 반환"""
        pass

    def validate_event(self, event_data: Dict[str, Any]) -> bool:
        """이벤트 데이터 유효성 검사"""
        return True

    def handle_error(self, error: Exception, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """에러 처리"""
        self.logger.error(f"이벤트 처리 실패: {error}")
        return None
