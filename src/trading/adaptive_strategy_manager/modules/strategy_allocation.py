"""
Strategy allocation module for adaptive strategy management.
Handles strategy allocation based on market regimes.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class StrategyAllocation:
    """
    Strategy allocation data class.
    
    Attributes:
        strategy_name: Name of the trading strategy
        weight: Allocation weight (0-1)
        confidence: Confidence score for this allocation
        expected_return: Expected return from this strategy
        risk_score: Risk score (0-100)
    """
    strategy_name: str
    weight: float
    confidence: float
    expected_return: float
    risk_score: float


class RegimeBasedAllocator:
    """Handles regime-based strategy allocation"""
    
    def __init__(self):
        # Regime-based strategy preferences
        self.regime_strategies: Dict[str, Dict[str, float]] = {
            'low_volatility': {
                'mean_reversion': 0.45,
                'market_making': 0.35,
                'arbitrage': 0.15,
                'momentum': 0.05
            },
            'normal': {
                'momentum': 0.35,
                'mean_reversion': 0.25,
                'market_making': 0.20,
                'arbitrage': 0.20
            },
            'high_volatility': {
                'momentum': 0.50,
                'arbitrage': 0.25,
                'mean_reversion': 0.15,
                'market_making': 0.10
            },
            'extreme_volatility': {
                'arbitrage': 0.60,
                'momentum': 0.25,
                'market_making': 0.10,
                'mean_reversion': 0.05
            },
            'crisis': {
                'arbitrage': 0.70,
                'market_making': 0.20,
                'momentum': 0.05,
                'mean_reversion': 0.05
            }
        }
        
        # Base expected returns (annualized)
        self.base_returns = {
            'momentum': 0.15,
            'mean_reversion': 0.12,
            'arbitrage': 0.08,
            'market_making': 0.10
        }
        
        # Base risk scores
        self.base_risks = {
            'momentum': 65,
            'mean_reversion': 45,
            'arbitrage': 25,
            'market_making': 35
        }
        
    def get_base_allocation(self, regime_name: str) -> Dict[str, float]:
        """Get base allocation for a given regime"""
        return self.regime_strategies.get(
            regime_name, 
            self.regime_strategies['normal']
        ).copy()
    
    def map_regime_to_name(self, regime: int) -> str:
        """Map regime number to name"""
        regime_names = ['low_volatility', 'normal', 'high_volatility', 'extreme_volatility']
        
        try:
            # Ensure valid regime index
            regime = max(0, min(regime, len(regime_names) - 1))
            return regime_names[regime]
        except (IndexError, TypeError) as e:
            logger.warning(f"Invalid regime {regime}, defaulting to normal: {e}")
            return 'normal'
    
    def apply_regime_confidence(self, allocation: Dict[str, float], 
                              regime_confidence: float,
                              min_confidence_threshold: float = 0.3) -> Dict[str, float]:
        """
        Apply regime confidence to allocation.
        Low confidence results in more balanced allocation.
        """
        try:
            # Validate regime confidence
            regime_confidence = max(0.0, min(1.0, regime_confidence))
            
            if regime_confidence < min_confidence_threshold:
                # Low confidence, use more balanced allocation
                adjusted = {}
                
                if not allocation:
                    return allocation
                
                balanced_weight = 1.0 / len(allocation)
                
                for strategy in allocation.keys():
                    # Blend between regime-based and balanced allocation
                    regime_weight = allocation.get(strategy, balanced_weight)
                    blended_weight = (
                        regime_weight * regime_confidence + 
                        balanced_weight * (1 - regime_confidence)
                    )
                    adjusted[strategy] = blended_weight
                    
                return adjusted
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error applying regime confidence: {e}")
            return allocation.copy()
    
    def estimate_expected_return(self, strategy: str, volatility: float = 0.02) -> float:
        """Estimate expected return for a strategy given market volatility"""
        try:
            base_return = self.base_returns.get(strategy, 0.1)
            
            # Validate volatility
            volatility = max(0.0, min(1.0, volatility))
            
            if strategy == 'momentum':
                # Momentum benefits from higher volatility
                volatility_factor = min(volatility / 0.02, 2.0)
                return base_return * volatility_factor
            elif strategy == 'arbitrage':
                # Arbitrage benefits from market inefficiencies
                volatility_factor = min(volatility / 0.015, 1.5)
                return base_return * volatility_factor
            
            return base_return
            
        except Exception as e:
            logger.error(f"Error estimating return for {strategy}: {e}")
            return 0.05  # Conservative default
    
    def calculate_strategy_risk(self, strategy: str, volatility: float = 0.02) -> float:
        """Calculate risk score for a strategy (0-100)"""
        try:
            base_risk = self.base_risks.get(strategy, 50)
            
            # Validate volatility
            volatility = max(0.0, min(1.0, volatility))
            volatility_multiplier = volatility / 0.02
            
            adjusted_risk = base_risk * volatility_multiplier
            
            return max(10, min(90, adjusted_risk))
            
        except Exception as e:
            logger.error(f"Error calculating risk for {strategy}: {e}")
            return 50  # Medium risk default
