import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class BTCRLTrader:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_data = None
        self.test_data = None
    
    def prepare_training_data(self, historical_data):
        """
        Prepare data for training
        """
        # Split data into train and test
        train_size = int(len(historical_data) * 0.8)
        self.train_data = historical_data[:train_size]
        self.test_data = historical_data[train_size:]
    
    def train(self):
        """
        Train RL model using PPO
        """
        from models.environments import TradingEnvironment
        
        # Create environment
        env = DummyVecEnv([lambda: TradingEnvironment(self.train_data.values)])
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=self.config.RL_PARAMS['learning_rate'],
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./ppo_trading_logs/"
        )
        
        # Train model
        self.model.learn(total_timesteps=100000)
    
    def predict_action(self, state):
        """
        Predict trading action
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        action, _ = self.model.predict(state)
        return action
    
    def save_model(self, path='btc_trading_model'):
        """
        Save trained model
        """
        self.model.save(path)
    
    def load_model(self, path='btc_trading_model'):
        """
        Load pre-trained model
        """
        self.model = PPO.load(path)
