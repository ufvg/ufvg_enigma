import gym
from gym import spaces
import numpy as np

class TestnetTradingEnv(gym.Env):
    def __init__(self, lob_data, window_size=10):
        self.lob_data = lob_data  # Shape: (n_samples, n_features)
        self.window_size = window_size
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 40))
        self.current_step = window_size
        self.position = 0  # 0=Flat, 1=Long, -1=Short

    def step(self, action):
        # Execute action
        self._take_action(action)
        
        # Update state
        self.current_step += 1
        next_state = self.lob_data[self.current_step - self.window_size : self.current_step]
        
        # Calculate reward (Testnet has no fees, but include slippage)
        reward = self._calculate_pnl() - abs(self.position) * 0.0001  # Simulate slippage
        
        done = self.current_step >= len(self.lob_data) - 1
        return next_state, reward, done, {}

    def _take_action(self, action):
        if action == 0:   # Long
            self.position = 1
        elif action == 1: # Short
            self.position = -1
        else:              # Flat
            self.position = 0