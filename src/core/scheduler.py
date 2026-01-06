
import time

def run_once(callback, interval_s: float = 0.1) -> None:
    callback()
    time.sleep(interval_s)
