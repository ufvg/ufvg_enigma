import numpy as np
import pandas as pd
import ta

class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df):
        """
        Add multiple technical indicators to DataFrame
        """
        # Moving Averages
        df['MA_10'] = df['close'].rolling(window=10).mean()
        df['MA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        
        # ATR for volatility
        df['ATR'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()
        
        return df.dropna()

