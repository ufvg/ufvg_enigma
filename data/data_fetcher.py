import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta  # Technical Analysis Library

class BTCDataFetcher:
    def __init__(self, symbol='BTC/USDT', timeframe='1m'):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Futures market
            }
        })
        self.symbol = symbol
        self.timeframe = timeframe
    
    def fetch_historical_btc_data(self, start_date, end_date, include_advanced_features=True):
        """
        Fetch comprehensive BTC historical data with advanced features
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD'
            end_date (str): End date in 'YYYY-MM-DD'
            include_advanced_features (bool): Add technical indicators
        
        Returns:
            pd.DataFrame: Comprehensive BTC OHLCV data
        """
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Fetch OHLCV data
        ohlcv = self.exchange.fetch_ohlcv(
            symbol=self.symbol, 
            timeframe=self.timeframe, 
            since=start_timestamp
        )
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        if include_advanced_features:
            # Advanced Technical Indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
        
        return df
