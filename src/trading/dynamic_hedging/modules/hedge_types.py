"""
Hedge Types Module

Defines hedge-related data structures and enumerations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, List
from datetime import datetime


class HedgeType(Enum):
    """Types of hedging strategies."""
    BETA_HEDGE = "beta_hedge"
    VOLATILITY_HEDGE = "volatility_hedge"
    CORRELATION_HEDGE = "correlation_hedge"
    TAIL_RISK_HEDGE = "tail_risk_hedge"
    CONCENTRATION_HEDGE = "concentration_hedge"
    DELTA_HEDGE = "delta_hedge"
    GAMMA_HEDGE = "gamma_hedge"
    VEGA_HEDGE = "vega_hedge"


class HedgeUrgency(Enum):
    """Hedge urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HedgeStatus(Enum):
    """Hedge position status."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class HedgeRecommendation:
    """
    Hedge recommendation structure.
    
    Attributes:
        hedge_type: Type of hedge (beta_hedge, volatility_hedge, etc.)
        symbol: Instrument symbol for hedging
        side: Trade side (buy/sell)
        size: Position size for hedge
        hedge_ratio: Ratio of hedge to portfolio
        reason: Explanation for hedge recommendation
        urgency: Urgency level (low/medium/high)
        expected_cost: Expected cost of hedge in base currency
        expected_protection: Expected protection value
    """
    hedge_type: str
    symbol: str
    side: str
    size: float
    hedge_ratio: float
    reason: str
    urgency: str
    expected_cost: float
    expected_protection: float
    
    # Optional fields
    target_metric: Optional[str] = None
    threshold_breached: Optional[float] = None
    confidence: float = 0.0
    expiry: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'hedge_type': self.hedge_type,
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'hedge_ratio': self.hedge_ratio,
            'reason': self.reason,
            'urgency': self.urgency,
            'expected_cost': self.expected_cost,
            'expected_protection': self.expected_protection,
            'target_metric': self.target_metric,
            'threshold_breached': self.threshold_breached,
            'confidence': self.confidence,
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'metadata': self.metadata
        }


@dataclass
class HedgePosition:
    """
    Active hedge position tracking.
    
    Attributes:
        hedge_id: Unique hedge identifier
        hedge_type: Type of hedge
        symbol: Hedging instrument symbol
        side: Position side
        size: Position size
        entry_price: Entry price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        status: Current status
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    hedge_id: str
    hedge_type: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    status: HedgeStatus
    created_at: datetime
    updated_at: datetime
    
    # Additional tracking fields
    hedge_ratio: float = 0.0
    expected_cost: float = 0.0
    expected_protection: float = 0.0
    actual_cost: float = 0.0
    actual_protection: float = 0.0
    reason: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current P&L."""
        self.current_price = current_price
        if self.side == 'buy':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # sell
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        return self.unrealized_pnl
    
    def effectiveness_ratio(self) -> float:
        """Calculate hedge effectiveness ratio."""
        if self.expected_protection == 0:
            return 0
        return self.actual_protection / self.expected_protection
    
    def cost_efficiency(self) -> float:
        """Calculate cost efficiency."""
        if self.actual_cost == 0:
            return 0
        return self.actual_protection / self.actual_cost


@dataclass
class HedgeInstrument:
    """
    Hedging instrument specification.
    
    Attributes:
        symbol: Instrument symbol
        instrument_type: Type (future, option, etc.)
        underlying: Underlying asset
        cost_bps: Cost in basis points
        effectiveness: Expected effectiveness (0-1)
        min_size: Minimum position size
        max_size: Maximum position size
    """
    symbol: str
    instrument_type: str
    underlying: str
    cost_bps: float
    effectiveness: float
    min_size: float
    max_size: float
    
    # Optional fields
    expiry: Optional[datetime] = None
    strike: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    liquidity_score: float = 1.0
    
    def is_available(self) -> bool:
        """Check if instrument is currently available."""
        if self.expiry and self.expiry < datetime.now():
            return False
        return self.liquidity_score > 0.5
    
    def calculate_cost(self, notional: float) -> float:
        """Calculate hedge cost for given notional."""
        return notional * self.cost_bps / 10000


@dataclass
class HedgePerformance:
    """
    Hedge performance metrics.
    
    Tracks the historical performance of hedging strategies.
    """
    hedge_id: str
    hedge_type: str
    start_date: datetime
    end_date: Optional[datetime]
    initial_portfolio_value: float
    protected_value: float
    hedge_cost: float
    realized_pnl: float
    max_drawdown_prevented: float
    effectiveness_score: float
    
    def calculate_roi(self) -> float:
        """Calculate return on hedge investment."""
        if self.hedge_cost == 0:
            return 0
        return (self.realized_pnl - self.hedge_cost) / self.hedge_cost
    
    def calculate_protection_ratio(self) -> float:
        """Calculate protection to cost ratio."""
        if self.hedge_cost == 0:
            return 0
        return self.protected_value / self.hedge_cost


@dataclass
class RiskMetrics:
    """
    Portfolio risk metrics for hedge analysis.
    """
    portfolio_beta: float
    portfolio_volatility: float
    value_at_risk: float
    conditional_var: float
    max_drawdown: float
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    concentration_scores: Optional[Dict[str, float]] = None
    stress_test_results: Optional[Dict[str, float]] = None
    
    def exceeds_threshold(self, metric: str, threshold: float) -> bool:
        """Check if a metric exceeds threshold."""
        value = getattr(self, metric, None)
        if value is None:
            return False
        return value > threshold
