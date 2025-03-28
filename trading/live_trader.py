import ccxt
import logging
import time

class BTCLiveTrader:
    def __init__(self, config, rl_agent):
        self.config = config
        self.agent = rl_agent
        
        # Binance Testnet setup
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            },
            'urls': {
                'api': ccxt.binance().urls['test']  # Testnet URL
            }
        })
        
        self.symbol = 'BTC/USDT'
        self.logger = logging.getLogger(__name__)
    
    def execute_trade(self, action):
        """
        Execute trade on Binance Testnet
        """
        try:
            # Get current market price
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Trade amount (1% of account balance)
            balance = self.exchange.fetch_balance()
            trade_amount = balance['USDT']['total'] * 0.01
            
            if action == 'buy':
                order = self.exchange.create_market_buy_order(
                    self.symbol, 
                    trade_amount / current_price
                )
                self.logger.info(f"Buy order executed: {order}")
            
            elif action == 'sell':
                order = self.exchange.create_market_sell_order(
                    self.symbol, 
                    trade_amount / current_price
                )
                self.logger.info(f"Sell order executed: {order}")
            
            time.sleep(5)  # Prevent rate limiting
        
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
    
    def calculate_trade_size(self, total_balance, risk_percentage=0.01):
        """
        Calculate appropriate trade size based on risk
        """
        return total_balance * risk_percentage
