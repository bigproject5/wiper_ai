from typing import Dict, Any, Optional
import logging
from app.events.event_handler import BaseEventHandler
from app.events.test_started_handler import TestStartedEventHandler

logger = logging.getLogger(__name__)

class EventRouter:
    """이벤트 라우터 - 단순화됨"""

    def __init__(self):
        self.handlers: Dict[str, BaseEventHandler] = {}
        self._register_handlers()

    def _register_handlers(self):
        """핸들러 등록"""
        test_started_handler = TestStartedEventHandler()
        self.handlers[test_started_handler.get_event_type()] = test_started_handler

        logger.info(f"등록된 핸들러: {list(self.handlers.keys())}")

    async def route_event(self, event_type: str, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """이벤트를 적절한 핸들러로 라우팅"""
        try:
            if event_type not in self.handlers:
                logger.warning(f"지원하지 않는 이벤트 타입: {event_type}")
                return None

            handler = self.handlers[event_type]
            logger.info(f"이벤트 처리 시작 - 타입: {event_type}")

            result = await handler.handle(event_data)

            if result:
                logger.info(f"이벤트 처리 완료 - 타입: {event_type}")

            return result

        except Exception as e:
            logger.error(f"이벤트 라우팅 실패 - 타입: {event_type}, 오류: {e}")
            return {"status": "error", "error": str(e)}

    def get_supported_events(self) -> list[str]:
        """지원하는 이벤트 타입 목록 반환"""
        return list(self.handlers.keys())
