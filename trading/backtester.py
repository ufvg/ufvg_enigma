import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class BTCBacktester:
    def __init__(self, rl_agent):
        self.agent = rl_agent
    
    def run_comprehensive_backtest(self, initial_balance=10000):
        """
        Run detailed backtest of trading strategy
        
        Returns:
            dict: Comprehensive backtest results
        """
        balance = initial_balance
        trades = []
        portfolio_values = [initial_balance]
        
        # Simulate trading
        for _ in range(len(self.agent.test_data)):
            action = self.agent.predict_action(self.agent.test_data[_])
            trade_result = self._simulate_trade(action, balance)
            
            balance = trade_result['new_balance']
            portfolio_values.append(balance)
            
            trades.append({
                'action': action,
                'price': trade_result['price'],
                'balance': balance
            })
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return_percentage': ((balance - initial_balance) / initial_balance) * 100,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'total_trades': len(trades),
            'winning_trades': sum(1 for t in trades if t['balance'] > initial_balance),
            'trades': trades
        }
    
    def _simulate_trade(self, action, current_balance):
        """
        Simulate trade based on action
        """
        # Mock trading logic
        price = np.random.uniform(9000, 11000)  # Random price simulation
        trade_amount = current_balance * 0.1  # 10% of balance
        
        if action == 'buy':
            new_balance = current_balance - trade_amount
        elif action == 'sell':
            new_balance = current_balance + trade_amount
        else:
            new_balance = current_balance
        
        return {
            'new_balance': new_balance,
            'price': price
        }
    
    def _calculate_max_drawdown(self, portfolio_values):
        """
        Calculate maximum drawdown
        """
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100
    
    def plot_performance(self):
        """
        Create performance visualization
        """
        plt.figure(figsize=(12, 6))
        plt.title('Portfolio Performance')
        plt.xlabel('Trading Steps')
        plt.ylabel('Portfolio Value')
        plt.tight_layout()
        plt.savefig('portfolio_performance.png')
        plt.close()
    
    def save_backtest_results(self, results, filename='backtest_results.json'):
        """
        Save backtest results to JSON
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
