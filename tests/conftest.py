
import pytest

@pytest.fixture
def sample_watchlist(tmp_path):
    path = tmp_path / 'watchlist.csv'
    path.write_text('symbol\nAAPL\n', encoding='utf-8')
    return str(path)
