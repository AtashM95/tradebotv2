
from src.core.session_manager import SessionManager

def test_session_manager():
    ctx = SessionManager('paper', True).start()
    assert ctx.mode == 'paper'
