import os
import yaml
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self._config = yaml.safe_load(file)
        
        # Binance Testnet Credentials
        self.BINANCE_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
        self.BINANCE_SECRET_KEY = os.getenv('BINANCE_TESTNET_SECRET_KEY')
        
        # Trading Specific
        self.TRADING_PARAMS = {
            'symbol': 'BTC/USDT',
            'timeframe': '1m',
            'leverage': 10,
            'max_trade_amount': 100,  # USDT
            'initial_balance': 10000,
            'trading_fee_rate': 0.0004  # 0.04% per trade
        }
        
        # RL Training Parameters
        self.RL_PARAMS = {
            'total_timesteps': 500000,
            'learning_rate': 0.0003,
            'batch_size': 64,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 10000
        }
    
    def get(self, key):
        return self._config.get(key)
