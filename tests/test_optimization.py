
from src.optimization.grid_search import GridSearch

def test_optimization():
    assert GridSearch().run()['status'] == 'ok'
