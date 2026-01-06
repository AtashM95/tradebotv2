
from src.ml.ml_pipeline import MlPipeline

def test_ml_pipeline():
    assert MlPipeline().run()['status'] == 'ok'
