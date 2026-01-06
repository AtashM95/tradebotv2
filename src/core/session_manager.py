
from datetime import datetime
from .contracts import RunContext

class SessionManager:
    def __init__(self, mode: str, dry_run: bool) -> None:
        self.mode = mode
        self.dry_run = dry_run

    def start(self) -> RunContext:
        run_id = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        return RunContext(run_id=run_id, mode=self.mode, timestamps={'start': datetime.utcnow()}, cost_budget=5.0, dry_run=self.dry_run)
