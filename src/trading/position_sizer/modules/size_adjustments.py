"""
Size adjustment module for position sizing.
Handles various adjustments and limits for calculated sizes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SizeAdjustments:
    """Handles position size adjustments and limits"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
    def apply_adjustments(self, base_size: float, signal: Dict[str, Any], 
                         market_data: Dict[str, float], current_drawdown: float,
                         correlation_factor: float) -> float:
        """
        Apply various adjustments to base position size with error handling.
        
        Args:
            base_size: Base position size
            signal: Trading signal
            market_data: Market conditions
            current_drawdown: Current portfolio drawdown
            correlation_factor: Correlation adjustment factor
            
        Returns:
            Adjusted position size
        """
        try:
            adjusted_size = base_size
            
            # Validate base size
            if not isinstance(adjusted_size, (int, float)) or adjusted_size <= 0:
                return self.params['min_position_size']
            
            # Confidence adjustment
            confidence = signal.get('confidence', 0.5)
            if 0 < confidence < self.params['confidence_threshold']:
                confidence_factor = confidence / self.params['confidence_threshold']
                adjusted_size *= confidence_factor
            
            # Correlation adjustment
            adjusted_size *= correlation_factor
            
            # Drawdown adjustment
            current_drawdown = min(current_drawdown, 1.0)
            if current_drawdown > 0.1:
                drawdown_factor = 1 - (current_drawdown - 0.1) * 2  # Reduce by 20% per 10% drawdown
                drawdown_factor = max(0.3, drawdown_factor)  # Minimum 30% of original
                adjusted_size *= drawdown_factor
            
            # Time of day adjustment (crypto markets can be less liquid at certain times)
            try:
                hour = pd.Timestamp.now().hour
                if 0 <= hour < 6:  # Late night US time
                    adjusted_size *= 0.8
            except Exception as e:
                logger.warning(f"Error applying time adjustment: {e}")
            
            # Market condition adjustments
            adjusted_size = self._apply_market_adjustments(adjusted_size, market_data)
            
            # Ensure valid result
            if not np.isfinite(adjusted_size) or adjusted_size <= 0:
                return self.params['min_position_size']
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error applying adjustments: {e}")
            return base_size if base_size > 0 else self.params['min_position_size']
    
    def _apply_market_adjustments(self, size: float, market_data: Dict[str, float]) -> float:
        """Apply market condition based adjustments"""
        try:
            # Volume adjustment
            volume_ratio = market_data.get('volume_ratio', 1.0)
            if volume_ratio < 0.5:  # Low volume
                size *= 0.7
            elif volume_ratio > 2.0:  # High volume
                size *= 1.1
                
            # Spread adjustment
            spread_pct = market_data.get('spread_pct', 0.001)
            if spread_pct > 0.005:  # Wide spread
                size *= 0.8
                
            # Volatility spike adjustment
            volatility_spike = market_data.get('volatility_spike', False)
            if volatility_spike:
                size *= 0.6
                
            return size
            
        except Exception as e:
            logger.error(f"Error in market adjustments: {e}")
            return size
    
    def calculate_correlation_adjustment(self, symbol: str, 
                                       positions: Dict[str, Any],
                                       get_correlation_func) -> float:
        """
        Calculate position size adjustment based on correlation with existing positions.
        
        Args:
            symbol: Trading symbol
            positions: Current positions
            get_correlation_func: Function to get correlation between symbols
            
        Returns:
            Correlation adjustment factor
        """
        try:
            if not positions:
                return 1.0
            
            # Calculate average correlation
            correlations = []
            for existing_symbol in positions:
                try:
                    correlation = get_correlation_func(symbol, existing_symbol)
                    if 0 <= correlation <= 1:
                        correlations.append(correlation)
                except Exception as e:
                    logger.warning(f"Error getting correlation: {e}")
                    continue
            
            if not correlations:
                return 1.0
                
            avg_correlation = np.mean(correlations)
            
            # Apply penalty for high correlation
            if avg_correlation > 0.7:
                return 1 - self.params['correlation_penalty']
            elif avg_correlation > 0.5:
                return 1 - self.params['correlation_penalty'] * 0.5
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return 1.0
    
    def apply_limits(self, size: float, entry_price: float, 
                    available_capital: float) -> float:
        """
        Apply position size limits with error handling.
        
        Args:
            size: Calculated position size
            entry_price: Entry price
            available_capital: Available capital
            
        Returns:
            Limited position size
        """
        try:
            # Validate inputs
            if not isinstance(size, (int, float)) or size <= 0:
                return self.params['min_position_size']
            if not isinstance(entry_price, (int, float)) or entry_price <= 0:
                return self.params['min_position_size']
            
            # Maximum position size
            if available_capital > 0:
                max_size = available_capital * self.params['max_position_pct']
                size = min(size, max_size)
            
            # Minimum position size
            size = max(size, self.params['min_position_size'])
            
            # Round to reasonable precision
            position_units = size / entry_price
            position_units = round(position_units, 8)  # 8 decimal places for crypto
            
            final_size = position_units * entry_price
            
            # Final validation
            if not np.isfinite(final_size) or final_size <= 0:
                return self.params['min_position_size']
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error applying limits: {e}")
            return self.params['min_position_size']
    
    def calculate_method_weights(self, market_data: Dict[str, float], 
                               portfolio_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate weights for different sizing methods based on market conditions with error handling.
        
        Args:
            market_data: Market conditions
            portfolio_state: Portfolio state
            
        Returns:
            Dictionary of method weights
        """
        try:
            weights = {
                'fixed_fractional': 0.2,
                'kelly': 0.15,
                'volatility_based': 0.15,
                'optimal_f': 0.1,
                'risk_parity': 0.1,
                'machine_learning': 0.15,
                'regime_adjusted': 0.15
            }
            
            # Adjust based on market regime
            regime = market_data.get('regime', 'normal')
            
            if regime == 'high_volatility':
                # Favor regime-adjusted and volatility-based in high volatility
                weights['regime_adjusted'] = 0.3
                weights['volatility_based'] = 0.25
                weights['fixed_fractional'] = 0.2
                weights['machine_learning'] = 0.15
                weights['kelly'] = 0.05
                weights['optimal_f'] = 0.025
                weights['risk_parity'] = 0.025
                
            elif regime == 'low_volatility':
                # Favor Kelly and machine learning in low volatility
                weights['kelly'] = 0.25
                weights['machine_learning'] = 0.25
                weights['optimal_f'] = 0.2
                weights['regime_adjusted'] = 0.15
                weights['volatility_based'] = 0.1
                weights['fixed_fractional'] = 0.05
            
            elif regime == 'trending':
                # Favor trend-following methods
                weights['machine_learning'] = 0.3
                weights['kelly'] = 0.2
                weights['regime_adjusted'] = 0.2
                weights['fixed_fractional'] = 0.15
                weights['volatility_based'] = 0.1
                weights['optimal_f'] = 0.025
                weights['risk_parity'] = 0.025
                
            # Adjust based on portfolio state
            position_count = len(portfolio_state.get('positions', {}))
            if position_count > 5:
                # More positions, favor risk parity
                weights['risk_parity'] = 0.25
                weights['fixed_fractional'] = 0.1
                
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                # Fallback to equal weights
                num_methods = len(weights)
                weights = {k: 1.0/num_methods for k in weights}
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating method weights: {e}")
            # Return equal weights as fallback
            return {
                'fixed_fractional': 1.0,
                'kelly': 0,
                'volatility_based': 0,
                'optimal_f': 0,
                'risk_parity': 0,
                'machine_learning': 0,
                'regime_adjusted': 0
            }
