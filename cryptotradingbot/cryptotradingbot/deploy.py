import ccxt
import time
from utils.data_fetcher import stream_testnet_lob
from utils.preprocessor import normalize_lob
import config
from models.attention_model import build_transformer_model
import numpy as np


# Load trained model
model = build_transformer_model(input_shape=(10, 40))
model.load_weights('models/testnet_transformer_agent.h5')

# Initialize Testnet exchange
exchange = ccxt.binance(config.BINANCE_TESTNET_CONFIG)

def execute_testnet_order(action):
    if action == 0:   # Buy BTC
        exchange.create_market_buy_order('BTC/USDT', 0.001)  # Testnet allows small orders
    elif action == 1: # Sell BTC
        exchange.create_market_sell_order('BTC/USDT', 0.001)
    else:             # Close position
        pass  # Adjust based on your position logic

while True:
    lob = stream_testnet_lob()
    state = normalize_lob(lob).reshape(1, 10, 40)  # Match training shape
    action = np.argmax(model.predict(state))
    execute_testnet_order(action)
    time.sleep(60)  # Trade every minute