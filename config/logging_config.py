
import json
import logging
from logging import StreamHandler

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'time': self.formatTime(record, self.datefmt),
        }
        if hasattr(record, 'run_id'):
            payload['run_id'] = record.run_id
        if record.exc_info:
            payload['error'] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = 'INFO') -> None:
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
