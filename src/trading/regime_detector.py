"""
Regime-Aware Position Sizing Module

Adjusts position sizes based on market regime and conditions.
This module provides dynamic position sizing that adapts to different
market environments and risk conditions.

Classes:
    RegimeAwarePositionSizer: Enhanced position sizing with regime adaptation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class RegimeAwarePositionSizer:
    """
    Enhanced position sizing that adapts to market regimes.
    
    This class implements regime-aware position sizing that adjusts
    position sizes based on:
    - Market regime (trending, sideways, volatile)
    - Volatility levels
    - Signal confidence
    - Market stress indicators
    - Portfolio correlations
    
    Attributes:
        risk_manager: Risk management instance
        regime_multipliers: Position size multipliers for each regime
        confidence_multipliers: Multipliers based on signal confidence
    """
    
    def __init__(self, risk_manager) -> None:
        """
        Initialize the Regime-Aware Position Sizer.
        
        Args:
            risk_manager: Risk management instance
        """
        self.risk_manager = risk_manager
        
        # Regime-specific multipliers
        self.regime_multipliers: Dict[str, Dict[str, float]] = {
            'trending': {
                'volatility_low': 1.2,
                'volatility_medium': 1.0,
                'volatility_high': 0.7
            },
            'sideways': {
                'volatility_low': 0.8,
                'volatility_medium': 0.6,
                'volatility_high': 0.4
            },
            'volatile': {
                'volatility_low': 0.9,
                'volatility_medium': 0.5,
                'volatility_high': 0.3
            }
        }
        
        # Confidence multipliers
        self.confidence_multipliers: Dict[float, float] = {
            0.9: 1.3,  # Very high confidence
            0.8: 1.2,  # High confidence
            0.7: 1.0,  # Medium confidence
            0.6: 0.8,  # Low confidence
            0.5: 0.6   # Very low confidence
        }
    
    def calculate_regime_aware_position_size(
        self,
        base_size: float,
        regime_info: Dict[str, Any],
        signal_strength: float,
        confidence: float,
        market_conditions: Dict[str, float]
    ) -> float:
        """
        Calculate position size adjusted for market regime.
        
        This method applies multiple adjustments to the base position size
        based on current market conditions and regime characteristics.
        
        Args:
            base_size: Base position size from standard calculator
            regime_info: Market regime information
            signal_strength: Signal strength (0-1)
            confidence: Signal confidence (0-1)
            market_conditions: Current market conditions
            
        Returns:
            Adjusted position size
        """
        try:
            # Start with base size
            adjusted_size = base_size
            
            # Apply regime multiplier
            regime_multiplier = self._get_regime_multiplier(
                regime_info, 
                market_conditions
            )
            adjusted_size *= regime_multiplier
            
            # Apply confidence multiplier
            confidence_multiplier = self._get_confidence_multiplier(confidence)
            adjusted_size *= confidence_multiplier
            
            # Apply signal strength multiplier
            signal_multiplier = 0.5 + (signal_strength * 0.5)  # Range: 0.5 to 1.0
            adjusted_size *= signal_multiplier
            
            # Apply market stress adjustment
            stress_multiplier = self._get_stress_multiplier(market_conditions)
            adjusted_size *= stress_multiplier
            
            # Apply correlation adjustment
            correlation_multiplier = self._get_correlation_multiplier(market_conditions)
            adjusted_size *= correlation_multiplier
            
            # Ensure position size respects limits
            adjusted_size = self._apply_position_limits(adjusted_size)
            
            logger.debug(
                f"Position sizing: base={base_size:.4f}, regime={regime_multiplier:.2f}, "
                f"confidence={confidence_multiplier:.2f}, signal={signal_multiplier:.2f}, "
                f"stress={stress_multiplier:.2f}, correlation={correlation_multiplier:.2f}, "
                f"final={adjusted_size:.4f}"
            )
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error in regime-aware position sizing: {e}")
            return base_size  # Return base size as fallback
    
    def _get_regime_multiplier(
        self, 
        regime_info: Dict[str, Any], 
        market_conditions: Dict[str, float]
    ) -> float:
        """
        Get position size multiplier based on market regime.
        
        Args:
            regime_info: Regime information
            market_conditions: Market conditions
            
        Returns:
            Regime-based multiplier
        """
        try:
            # Determine regime type
            regime_type = regime_info.get('regime_type', 'sideways')
            if regime_type not in self.regime_multipliers:
                regime_type = 'sideways'
            
            # Determine volatility level
            volatility = market_conditions.get('volatility', 0.02)
            if volatility < 0.15:
                vol_level = 'volatility_low'
            elif volatility < 0.25:
                vol_level = 'volatility_medium'
            else:
                vol_level = 'volatility_high'
            
            multiplier = self.regime_multipliers[regime_type][vol_level]
            
            # Additional adjustments based on trend strength
            trend_strength = regime_info.get('trend_strength', 0.0)
            if regime_type == 'trending' and trend_strength > 0.7:
                multiplier *= 1.1  # Boost for strong trends
            elif regime_type == 'sideways' and trend_strength < 0.3:
                multiplier *= 0.9  # Reduce for weak trends in sideways market
            
            return multiplier
            
        except Exception as e:
            logger.error(f"Error calculating regime multiplier: {e}")
            return 1.0
    
    def _get_confidence_multiplier(self, confidence: float) -> float:
        """
        Get multiplier based on signal confidence.
        
        Args:
            confidence: Signal confidence (0-1)
            
        Returns:
            Confidence-based multiplier
        """
        try:
            # Find closest confidence level
            confidence_levels = sorted(
                self.confidence_multipliers.keys(), 
                reverse=True
            )
            
            for level in confidence_levels:
                if confidence >= level:
                    return self.confidence_multipliers[level]
            
            # Default to lowest confidence
            return self.confidence_multipliers[0.5]
            
        except Exception as e:
            logger.error(f"Error calculating confidence multiplier: {e}")
            return 1.0
    
    def _get_stress_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """
        Get multiplier based on market stress.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Stress-based multiplier
        """
        try:
            market_stress = market_conditions.get('market_stress', 0.0)
            
            # Reduce position size during high stress
            if market_stress > 0.8:
                return 0.5  # High stress
            elif market_stress > 0.6:
                return 0.7  # Medium stress
            elif market_stress > 0.4:
                return 0.9  # Low stress
            else:
                return 1.0  # Normal conditions
                
        except Exception as e:
            logger.error(f"Error calculating stress multiplier: {e}")
            return 1.0
    
    def _get_correlation_multiplier(
        self, 
        market_conditions: Dict[str, float]
    ) -> float:
        """
        Get multiplier based on portfolio correlation.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Correlation-based multiplier
        """
        try:
            market_correlation = market_conditions.get('market_correlation', 0.5)
            
            # Reduce position size when correlations are high
            if market_correlation > 0.8:
                return 0.7  # High correlation - reduce diversification benefit
            elif market_correlation > 0.6:
                return 0.85  # Medium correlation
            else:
                return 1.0  # Low correlation - normal sizing
                
        except Exception as e:
            logger.error(f"Error calculating correlation multiplier: {e}")
            return 1.0
    
    def _apply_position_limits(self, size: float) -> float:
        """
        Apply position size limits.
        
        Args:
            size: Calculated position size
            
        Returns:
            Limited position size
        """
        try:
            # Get risk parameters
            risk_params = self.risk_manager.risk_params
            
            # Maximum position size (% of capital)
            max_position_pct = risk_params.get('max_position_size', 0.15)
            current_capital = self.risk_manager.current_capital
            max_position_value = current_capital * max_position_pct
            
            # Assume average price of $50k for sizing (would use actual price in practice)
            avg_price = 50000
            max_size = max_position_value / avg_price
            
            # Apply maximum
            size = min(size, max_size)
            
            # Minimum position size
            min_size = risk_params.get('min_position_size', 0.001)
            size = max(size, min_size) if size > 0 else 0
            
            return size
            
        except Exception as e:
            logger.error(f"Error applying position limits: {e}")
            return size
    
    def calculate_kelly_adjusted_size(
        self,
        base_size: float,
        win_probability: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion adjusted position size.
        
        Args:
            base_size: Base position size
            win_probability: Probability of winning trade
            avg_win: Average win amount
            avg_loss: Average loss amount
            
        Returns:
            Kelly-adjusted position size
        """
        try:
            if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:
                return base_size
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_probability, q = 1-p
            b = avg_win / abs(avg_loss)
            p = win_probability
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly fraction to reasonable limits
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25%
            
            # Apply Kelly fraction to base size
            kelly_size = base_size * kelly_fraction / 0.25  # Normalize
            
            return kelly_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly adjusted size: {e}")
            return base_size
    
    def calculate_volatility_adjusted_size(
        self,
        base_size: float,
        current_volatility: float,
        target_volatility: float = 0.15
    ) -> float:
        """
        Adjust position size for volatility targeting.
        
        Args:
            base_size: Base position size
            current_volatility: Current market volatility
            target_volatility: Target volatility level
            
        Returns:
            Volatility-adjusted position size
        """
        try:
            if current_volatility <= 0:
                return base_size
            
            # Volatility scaling factor
            vol_scaling = target_volatility / current_volatility
            
            # Cap scaling to reasonable range
            vol_scaling = max(0.2, min(vol_scaling, 3.0))
            
            return base_size * vol_scaling
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjusted size: {e}")
            return base_size
    
    def get_position_sizing_metrics(self) -> Dict[str, Any]:
        """
        Get current position sizing metrics and statistics.
        
        Returns:
            Dictionary of position sizing metrics
        """
        try:
            positions = self.risk_manager.positions
            current_capital = self.risk_manager.current_capital
            
            if not positions:
                return {}
            
            # Calculate position statistics
            position_values = []
            position_weights = []
            
            for symbol, position in positions.items():
                current_price = position.get('current_price', position['entry_price'])
                value = position['size'] * current_price
                weight = value / current_capital if current_capital > 0 else 0
                
                position_values.append(value)
                position_weights.append(weight)
            
            metrics = {
                'total_positions': len(positions),
                'total_position_value': sum(position_values),
                'capital_utilization': (
                    sum(position_values) / current_capital 
                    if current_capital > 0 else 0
                ),
                'average_position_weight': (
                    np.mean(position_weights) if position_weights else 0
                ),
                'max_position_weight': max(position_weights) if position_weights else 0,
                'min_position_weight': min(position_weights) if position_weights else 0,
                'position_weight_std': (
                    np.std(position_weights) if position_weights else 0
                ),
                'concentration_ratio': (
                    max(position_weights) / sum(position_weights) 
                    if position_weights else 0
                )
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating position sizing metrics: {e}")
            return {}
    
    def suggest_optimal_position_count(
        self, 
        available_capital: float,
        avg_position_size: float,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Suggest optimal number of positions for diversification.
        
        Args:
            available_capital: Available capital for trading
            avg_position_size: Average position size
            correlation_matrix: Optional correlation matrix
            
        Returns:
            Suggested optimal position count
        """
        try:
            # Basic calculation based on capital
            basic_count = int(available_capital / avg_position_size)
            
            # Adjust for correlation if available
            if correlation_matrix is not None:
                # Higher correlation means need more positions for diversification
                avg_correlation = correlation_matrix.values[
                    np.triu_indices_from(correlation_matrix.values, k=1)
                ].mean()
                
                if avg_correlation > 0.7:
                    correlation_adjustment = 1.5  # Need more positions
                elif avg_correlation > 0.5:
                    correlation_adjustment = 1.2
                else:
                    correlation_adjustment = 1.0
                
                basic_count = int(basic_count * correlation_adjustment)
            
            # Apply practical limits
            optimal_count = max(3, min(basic_count, 15))  # Between 3 and 15 positions
            
            return optimal_count
            
        except Exception as e:
            logger.error(f"Error suggesting optimal position count: {e}")
            return 5  # Default
    
    def update_regime_parameters(self, new_params: Dict[str, Any]) -> None:
        """
        Update regime-specific parameters.
        
        Args:
            new_params: Dictionary of new parameters
        """
        try:
            if 'regime_multipliers' in new_params:
                self.regime_multipliers.update(new_params['regime_multipliers'])
            
            if 'confidence_multipliers' in new_params:
                self.confidence_multipliers.update(new_params['confidence_multipliers'])
            
            logger.info("Regime parameters updated")
            
        except Exception as e:
            logger.error(f"Error updating regime parameters: {e}")


# Example usage and testing
if __name__ == '__main__':
    # Mock risk manager for testing
    class MockRiskManager:
        def __init__(self):
            self.current_capital = 100000
            self.risk_params = {
                'max_position_size': 0.15,
                'min_position_size': 0.001
            }
            self.positions = {
                'BTC-USD': {
                    'size': 0.5,
                    'entry_price': 45000,
                    'current_price': 47000,
                    'side': 'long'
                }
            }
    
    # Test regime-aware position sizer
    risk_manager = MockRiskManager()
    sizer = RegimeAwarePositionSizer(risk_manager)
    
    # Test position sizing
    base_size = 0.1
    regime_info = {
        'regime_type': 'trending',
        'trend_strength': 0.8
    }
    market_conditions = {
        'volatility': 0.18,
        'market_stress': 0.3,
        'market_correlation': 0.6
    }
    
    adjusted_size = sizer.calculate_regime_aware_position_size(
        base_size=base_size,
        regime_info=regime_info,
        signal_strength=0.7,
        confidence=0.8,
        market_conditions=market_conditions
    )
    
    print(f"Base size: {base_size}")
    print(f"Adjusted size: {adjusted_size:.4f}")
    print(f"Sizing metrics: {sizer.get_position_sizing_metrics()}")