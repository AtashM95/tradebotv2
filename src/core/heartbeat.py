
from datetime import datetime

def heartbeat() -> dict:
    return {'status': 'ok', 'timestamp': datetime.utcnow().isoformat()}
