# scripts/backtest.py
import sys
import os
import logging
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.rl_agent import BTCRLTrader
from trading.backtester import BTCBacktester
from data.data_fetcher import BTCDataFetcher

def run_backtest():
    """
    Script to run comprehensive backtest on trained model
    """
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

        # Load pre-trained model
        logger.info("Loading pre-trained BTC trading model...")
        btc_trader = BTCRLTrader(config)
        btc_trader.load_model('btc_trading_model')
        
        # Run comprehensive backtest
        logger.info("Running comprehensive backtest...")
        backtester = BTCBacktester(btc_trader)
        backtest_results = backtester.run_comprehensive_backtest()
        
        # Visualize and save results
        backtester.plot_performance()
        backtester.save_backtest_results(backtest_results, 'detailed_backtest_results.json')
        
        logger.info("Backtest completed successfully!")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    run_backtest()
