import sys
import os
import logging
import time

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.rl_agent import BTCRLTrader
from trading.live_trader import BTCLiveTrader
from data.data_fetcher import BTCDataFetcher

def start_live_trading():
    """
    Script to start live trading on Binance Testnet
    """
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = Config()
        
        # Load pre-trained model
        logger.info("Loading pre-trained BTC trading model...")
        btc_trader = BTCRLTrader(config)
        btc_trader.load_model('btc_trading_model')
        
        # Initialize live trader
        logger.info("Initializing live trading...")
        live_trader = BTCLiveTrader(config, btc_trader)
        
        # Start continuous trading loop
        logger.info("Starting live trading on Binance Testnet...")
        while True:
            try:
                # Fetch latest market data
                latest_data = BTCDataFetcher().fetch_latest_btc_data()
                
                # Make trading decision
                trading_decision = btc_trader.predict_action(latest_data)
                
                # Execute trade
                live_trader.execute_trade(trading_decision)
                
                # Wait before next iteration
                time.sleep(60)  # 1-minute interval
                
            except Exception as trade_error:
                logger.error(f"Trading error: {trade_error}")
                time.sleep(60)  # Wait before retry

    except Exception as e:
        logger.error(f"Live trading initialization failed: {e}")
        raise

if __name__ == "__main__":
    start_live_trading()
