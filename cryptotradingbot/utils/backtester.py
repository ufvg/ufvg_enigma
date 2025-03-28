# utils/backtester.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import INITIAL_BALANCE, TRANSACTION_COST_RATE

class Backtester:
    def __init__(self, model, lob_data, initial_balance=INITIAL_BALANCE):
        self.model = model
        self.lob_data = lob_data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.portfolio_values = []
        self.trades = []
        self.current_step = 0
        self.position = 'flat'
        self.entry_price = None
        
    def _get_current_price(self):
        """Get mid price from LOB data"""
        bids = self.lob_data[self.current_step]['bids']
        asks = self.lob_data[self.current_step]['asks']
        return (bids[0][0] + asks[0][0]) / 2
    
    def _execute_trade(self, action, price):
        """Simulate trade execution with slippage and fees"""
        # Calculate slippage (0.1% of spread)
        spread = asks[0][0] - bids[0][0]
        slippage = spread * 0.001
        executed_price = price + (slippage if action == 'buy' else -slippage)
        
        # Calculate fees
        fee_rate = TRANSACTION_COST_RATE
        
        if action == 'buy':
            max_btc = self.balance / (executed_price * (1 + fee_rate))
            self.btc_held += max_btc
            self.balance -= max_btc * executed_price * (1 + fee_rate)
            self.position = 'long'
            
        elif action == 'sell':
            self.balance += self.btc_held * executed_price * (1 - fee_rate)
            self.btc_held = 0
            self.position = 'short'
            
        elif action == 'close':
            if self.position == 'long':
                self.balance += self.btc_held * executed_price * (1 - fee_rate)
                self.btc_held = 0
            self.position = 'flat'
            
        self.trades.append({
            'step': self.current_step,
            'action': action,
            'price': executed_price,
            'balance': self.balance,
            'btc_held': self.btc_held
        })
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        metrics = {
            'final_balance': self.balance + self.btc_held * self._get_current_price(),
            'sharpe_ratio': self._annualized_sharpe(returns),
            'max_drawdown': self._max_drawdown(),
            'win_rate': self._calculate_win_rate(),
            'total_trades': len(self.trades)
        }
        return metrics
    
    def _annualized_sharpe(self, returns):
        if len(returns) < 2:
            return 0
        return np.sqrt(365*24*60) * returns.mean() / returns.std()
    
    def _max_drawdown(self):
        peak = self.portfolio_values[0]
        max_dd = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
    
    def _calculate_win_rate(self):
        if len(self.trades) < 2:
            return 0
        profitable = 0
        for i in range(1, len(self.trades)):
            if self.trades[i]['balance'] > self.trades[i-1]['balance']:
                profitable += 1
        return profitable / len(self.trades)
    
    def run_backtest(self, window_size=10):
        """Run full backtest on historical data"""
        self.reset()
        
        for self.current_step in tqdm(range(window_size, len(self.lob_data))):
            # Get state window
            state = self.lob_data[self.current_step-window_size:self.current_step]
            
            # Predict action
            action_probs = self.model.predict(state.reshape(1, window_size, -1))
            action = np.argmax(action_probs)
            
            # Map action to trade
            current_price = self._get_current_price()
            if action == 0:  # Long
                if self.position != 'long':
                    if self.position == 'short':
                        self._execute_trade('close', current_price)
                    self._execute_trade('buy', current_price)
            elif action == 1:  # Short
                if self.position != 'short':
                    if self.position == 'long':
                        self._execute_trade('close', current_price)
                    self._execute_trade('sell', current_price)
            else:  # Flat
                if self.position != 'flat':
                    self._execute_trade('close', current_price)
            
            # Update portfolio value
            portfolio_value = self.balance + self.btc_held * current_price
            self.portfolio_values.append(portfolio_value)
            
        return self._calculate_metrics()
    
    def plot_results(self):
        """Generate performance visualizations"""
        plt.figure(figsize=(15, 8))
        
        # Equity curve
        plt.subplot(2, 2, 1)
        plt.plot(self.portfolio_values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Step')
        plt.ylabel('Value (USDT)')
        
        # Drawdown curve
        plt.subplot(2, 2, 2)
        rolling_max = pd.Series(self.portfolio_values).cummax()
        drawdown = (rolling_max - pd.Series(self.portfolio_values)) / rolling_max
        plt.plot(drawdown)
        plt.title('Maximum Drawdown')
        plt.xlabel('Step')
        plt.ylabel('Drawdown %')
        
        # Trade distribution
        plt.subplot(2, 2, 3)
        actions = [t['action'] for t in self.trades]
        pd.Series(actions).value_counts().plot(kind='bar')
        plt.title('Trade Distribution')
        
        # Daily returns
        plt.subplot(2, 2, 4)
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        plt.hist(returns, bins=50)
        plt.title('Returns Distribution')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    from models.attention_model import build_transformer_model
    import numpy as np
    
    # Load sample data
    lob_data = np.load('data/processed/testnet_lob.npy')
    model = build_transformer_model(input_shape=(10, 40))
    model.load_weights('models/testnet_transformer_agent.h5')
    
    # Run backtest
    backtester = Backtester(model, lob_data)
    metrics = backtester.run_backtest()
    print("\nBacktest Metrics:")
    for k, v in metrics.items():
        print(f"{k:15}: {v:.2f}")
    
    # Show plots
    backtester.plot_results()