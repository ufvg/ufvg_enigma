import unittest
import numpy as np
from models.environment import TradingEnvironment

class TestTradingEnvironment(unittest.TestCase):
    def setUp(self):
        # Create mock data
        mock_data = np.random.rand(1000, 5)
        self.env = TradingEnvironment(mock_data)

    def test_environment_reset(self):
        obs, _ = self.env.reset()
        self.assertIsNotNone(obs)
        self.assertEqual(self.env.current_step, 0)

    def test_trading_steps(self):
        obs, _ = self.env.reset()
        
        for _ in range(10):
            action = self.env.action_space.sample()
            next_obs, reward, done, _, _ = self.env.step(action)
            
            self.assertIsNotNone(next_obs)
            self.assertIsInstance(reward, float)
            self.assertIn(done, [True, False])