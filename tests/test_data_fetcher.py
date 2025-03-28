import unittest
import pandas as pd
from data.data_fetcher import BTCDataFetcher

class TestBTCDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = BTCDataFetcher()

    def test_fetch_historical_data(self):
        data = self.fetcher.fetch_historical_btc_data('2023-01-01', '2023-02-01')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue(len(data) > 0)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)

    def test_data_features(self):
        data = self.fetcher.fetch_historical_btc_data('2023-01-01', '2023-02-01')
        
        self.assertIn('rsi', data.columns)
        self.assertIn('macd', data.columns)
        self.assertIn('atr', data.columns)