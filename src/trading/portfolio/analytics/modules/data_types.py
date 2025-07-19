"""
Data types for portfolio analytics
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    var_95: float
    cvar_95: float
    diversification_ratio: float
    concentration_risk: float


@dataclass
class RebalancingRecommendation:
    """Rebalancing recommendation for a specific asset"""
    symbol: str
    current_weight: float
    target_weight: float
    weight_deviation: float
    action: str  # 'buy', 'sell', 'hold'
    amount_to_trade: float
    urgency: str  # 'low', 'medium', 'high'
    reason: str
