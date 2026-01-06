
from src.portfolio.portfolio_manager import PortfolioManager

def test_portfolio_manager():
    assert PortfolioManager().run()['status'] == 'ok'
