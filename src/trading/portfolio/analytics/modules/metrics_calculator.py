"""
Portfolio metrics calculation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from scipy import stats

from .data_types import PortfolioMetrics

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates portfolio performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self, 
                         returns: pd.Series, 
                         benchmark_returns: Optional[pd.Series] = None,
                         positions: Dict = None) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Risk-adjusted returns
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = self._calculate_sharpe_ratio(returns, excess_returns)
            sortino_ratio = self._calculate_sortino_ratio(returns, excess_returns)
            
            # Drawdown analysis
            max_drawdown = self._calculate_max_drawdown(returns)
            calmar_ratio = (total_return / max_drawdown) if max_drawdown > 0 else 0
            
            # Trade-based metrics
            win_rate, profit_factor = self._calculate_trade_metrics(positions)
            
            # Market-relative metrics
            beta, alpha, information_ratio = self._calculate_market_metrics(
                returns, benchmark_returns
            )
            
            # Value at Risk
            var_95, cvar_95 = self._calculate_var_metrics(returns)
            
            # Portfolio concentration
            diversification_ratio, concentration_risk = self._calculate_concentration_metrics(
                positions
            )
            
            return PortfolioMetrics(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                volatility=volatility,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                var_95=var_95,
                cvar_95=cvar_95,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, excess_returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() > 0:
            return excess_returns.mean() / returns.std() * np.sqrt(252)
        return 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, excess_returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                return excess_returns.mean() / downside_deviation * np.sqrt(252)
        return 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdowns.min())
    
    def _calculate_trade_metrics(self, positions: Dict) -> tuple:
        """Calculate trade-based metrics"""
        win_rate = 0
        profit_factor = 0
        
        if positions:
            closed_trades = [pos for pos in positions.values() if pos.get('status') == 'closed']
            if closed_trades:
                wins = sum(1 for trade in closed_trades if trade.get('pnl', 0) > 0)
                win_rate = wins / len(closed_trades)
                
                gross_profit = sum(trade.get('pnl', 0) for trade in closed_trades if trade.get('pnl', 0) > 0)
                gross_loss = abs(sum(trade.get('pnl', 0) for trade in closed_trades if trade.get('pnl', 0) < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return win_rate, profit_factor
    
    def _calculate_market_metrics(self, returns: pd.Series, 
                                 benchmark_returns: Optional[pd.Series]) -> tuple:
        """Calculate market-relative metrics"""
        beta = 0
        alpha = 0
        information_ratio = 0
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Beta calculation
            covariance = np.cov(returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha calculation (Jensen's alpha)
            expected_return = self.risk_free_rate / 252 + beta * (benchmark_returns.mean() - self.risk_free_rate / 252)
            alpha = (returns.mean() - expected_return) * 252  # Annualized
            
            # Information ratio
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
        
        return beta, alpha, information_ratio
    
    def _calculate_var_metrics(self, returns: pd.Series) -> tuple:
        """Calculate Value at Risk metrics"""
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        return var_95, cvar_95
    
    def _calculate_concentration_metrics(self, positions: Dict) -> tuple:
        """Calculate portfolio concentration metrics"""
        diversification_ratio = 0
        concentration_risk = 0
        
        if positions:
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            if total_value > 0:
                weights = [pos.get('value', 0) / total_value for pos in positions.values()]
                # Herfindahl-Hirschman Index for concentration
                concentration_risk = sum(w**2 for w in weights)
                # Simple diversification measure
                diversification_ratio = 1 / concentration_risk if concentration_risk > 0 else 0
        
        return diversification_ratio, concentration_risk
