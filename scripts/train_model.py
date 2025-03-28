import sys
import os
import logging
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.data_fetcher import BTCDataFetcher
from models.rl_agent import BTCRLTrader
from trading.backtester import BTCBacktester

def train_model():
    """
    Script to train the BTC trading model
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = Config()
        
        # Fetch historical data
        logger.info("Fetching historical BTC data...")
        data_fetcher = BTCDataFetcher()
        historical_data = data_fetcher.fetch_historical_btc_data(
            start_date='2022-01-01', 
            end_date='2024-01-01'
        )

        # Initialize and train RL agent
        logger.info("Initializing BTC trading agent...")
        btc_trader = BTCRLTrader(config)
        btc_trader.prepare_training_data(historical_data)
        
        logger.info("Starting model training...")
        btc_trader.train()
        
        # Backtest the model
        logger.info("Running backtest...")
        backtester = BTCBacktester(btc_trader)
        backtest_results = backtester.run_comprehensive_backtest()
        
        # Save model and results
        btc_trader.save_model('btc_trading_model')
        backtester.save_backtest_results(backtest_results, 'backtest_results.json')
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()


