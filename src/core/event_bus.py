
from collections import defaultdict
from typing import Callable, Any

class EventBus:
    def __init__(self) -> None:
        self.handlers = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        self.handlers[event].append(handler)

    def publish(self, event: str, payload: Any) -> None:
        for handler in self.handlers.get(event, []):
            handler(payload)
