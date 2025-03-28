import ccxt
import json
import threading
import numpy as np
from websocket import create_connection
from config import BINANCE_TESTNET_CONFIG, SYMBOL, LOB_LEVELS
from utils.preprocessor import process_lob
import time

def fetch_historical_lob(symbol=SYMBOL, limit=1000):
    """Fetch historical order book data from Binance Testnet"""
    exchange = ccxt.binance({
        **BINANCE_TESTNET_CONFIG,
        'enableRateLimit': True
    })
    
    try:
        orderbook = exchange.fetch_order_book(symbol, limit=limit)
        return {
            'bids': np.array(orderbook['bids'], dtype=float)[:LOB_LEVELS],
            'asks': np.array(orderbook['asks'], dtype=float)[:LOB_LEVELS],
            'timestamp': orderbook['timestamp']
        }
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def stream_testnet_lob(symbol=SYMBOL, callback=process_lob):
    """Stream real-time LOB data from Testnet WebSocket"""
    symbol_lower = symbol.replace('/', '').lower()
    ws_url = f"wss://testnet.binance.vision/ws/{symbol_lower}@depth20@100ms"
    
    def _websocket_thread():
        ws = None
        while True:
            try:
                ws = create_connection(ws_url)
                print(f"Connected to {ws_url}")
                
                while True:
                    data = json.loads(ws.recv())
                    # Add timestamp and structure data
                    structured_data = {
                        'bids': np.array(data['bids'], dtype=float)[:LOB_LEVELS],
                        'asks': np.array(data['asks'], dtype=float)[:LOB_LEVELS],
                        'timestamp': data['E']
                    }
                    if callback:
                        callback(structured_data)
                        
            except Exception as e:
                print(f"WebSocket error: {e}")
                if ws:
                    ws.close()
                time.sleep(5)  # Reconnect after 5 seconds
                
    # Start WebSocket in daemon thread
    thread = threading.Thread(target=_websocket_thread, daemon=True)
    thread.start()
    return thread

def get_testnet_balance():
    """Check available Testnet balance"""
    exchange = ccxt.binance(BINANCE_TESTNET_CONFIG)
    try:
        balance = exchange.fetch_balance()
        return {
            'USDT': balance['USDT']['free'],
            'BTC': balance['BTC']['free']
        }
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return None

if __name__ == "__main__":
    # Test the functions
    print("Testing historical data fetch...")
    lob = fetch_historical_lob()
    print(f"First bid: {lob['bids'][0] if lob else 'Failed'}")
    
    print("\nStarting WebSocket stream...")
    stream_thread = stream_testnet_lob()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped by user")