
from src.ui.dashboard import Dashboard

def test_dashboard():
    assert 'ready' in Dashboard().render()
