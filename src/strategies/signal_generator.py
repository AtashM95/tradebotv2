
from ..core.contracts import SignalIntent

class SignalGenerator:
    def combine(self, signals: list[SignalIntent]) -> SignalIntent | None:
        if not signals:
            return None
        best = max(signals, key=lambda s: s.confidence)
        return best
