from config.config import Config
from data.data_fetcher import BTCDataFetcher
from models.rl_agent import BTCRLTrader
from trading.backtester import BTCBacktester
from trading.live_trader import BTCLiveTrader
import logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load Configuration
    config = Config()
    
    try:
        # 1. Data Fetching
        logger.info("Fetching Historical BTC Data...")
        data_fetcher = BTCDataFetcher()
        historical_data = data_fetcher.fetch_historical_btc_data(
            '2022-01-01', 
            '2024-01-01'
        )
        
        # 2. Training RL Agent
        logger.info("Training BTC Trading Agent...")
        btc_trader = BTCRLTrader(config)
        btc_trader.prepare_training_data(historical_data)
        btc_trader.train()
        
        # 3. Backtesting
        logger.info("Running Backtest...")
        backtester = BTCBacktester(btc_trader)
        backtest_results = backtester.run_comprehensive_backtest()
        backtester.generate_performance_report(backtest_results)
        
        # 4. Live Trading Setup
        logger.info("Preparing Live Trading...")
        live_trader = BTCLiveTrader(config, btc_trader)
        live_trader.initialize_testnet_trading()
        
        logger.info("BTC Trading Bot Setup Complete!")
    
    except Exception as e:
        logger.error(f"Critical Error: {e}")

if __name__ == "__main__":
    main()
