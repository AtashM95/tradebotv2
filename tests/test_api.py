
from src.api.serializers import ok

def test_api_ok():
    assert ok('x')['status'] == 'ok'
