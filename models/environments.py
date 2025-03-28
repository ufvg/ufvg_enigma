import gymnasium as gym
import numpy as np

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Action space: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: price features + portfolio status
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(data.shape[1] + 3,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0
        self.total_reward = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute trading action
        """
        current_price = self.data[self.current_step, 0]  # Closing price
        
        # Trade execution logic
        if action == 1 and self.balance > current_price:  # Buy
            self.btc_held = self.balance / current_price
            self.balance = 0
        elif action == 2 and self.btc_held > 0:  # Sell
            self.balance = self.btc_held * current_price
            self.btc_held = 0
        
        # Calculate reward
        reward = self._calculate_reward(current_price)
        
        self.current_step += 1
        done = (self.current_step >= len(self.data) - 1)
        
        return self._get_observation(), reward, done, False, {}
    
    def _calculate_reward(self, current_price):
        """
        Calculate trading reward
        """
        # Simple reward based on portfolio value change
        total_value = self.balance + (self.btc_held * current_price)
        reward = (total_value - self.initial_balance) / self.initial_balance
        return reward
    
    def _get_observation(self):
        """
        Get current trading state
        """
        portfolio_info = [
            self.balance / self.initial_balance,
            self.btc_held,
            self.total_reward
        ]
        return np.concatenate([
            self.data[self.current_step], 
            portfolio_info
        ])
