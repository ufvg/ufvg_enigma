# utils/performance_metrics.py
import numpy as np
import pandas as pd

class PerformanceMetrics:
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.02):
        """
        Calculate Sharpe Ratio
        """
        return (np.mean(returns) - risk_free_rate) / np.std(returns)
    
    @staticmethod
    def max_drawdown(portfolio_values):
        """
        Calculate Maximum Drawdown
        """
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100
    
    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.02):
        """
        Calculate Sortino Ratio
        """
        negative_returns = returns[returns < 0]
        return (np.mean(returns) - risk_free_rate) / np.std(negative_returns)

