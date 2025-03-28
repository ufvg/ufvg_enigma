import unittest
from trading.live_trader import BTCLiveTrader
from config.config import Config

class TestTradingLogic(unittest.TestCase):
    def setUp(self):
        config = Config()
        # Mock RL Trader
        self.live_trader = BTCLiveTrader(config, None)

    def test_risk_management(self):
        # Test trade sizing and risk limits
        trade_amount = self.live_trader.calculate_trade_size(100000, 0.02)
        self.assertTrue(0 <= trade_amount <= 2000)

    def test_trade_execution(self):
        # Simulate trade execution
        result = self.live_trader.simulate_trade('buy', 100)
        self.assertTrue(result in ['success', 'insufficient_balance', 'market_closed'])

if __name__ == '__main__':
    unittest.main()
