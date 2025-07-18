"""
File: position_sizer.py
Modified: 2024-12-19
Refactored: 2025-07-18

Advanced position sizing with modular architecture.
This file coordinates the position sizing modules for better maintainability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import traceback

# Import position sizing modules
from modules.base_sizing import BaseSizingMethods
from modules.advanced_sizing import AdvancedSizingMethods
from modules.size_adjustments import SizeAdjustments

from utils.logger import setup_logger

logger = setup_logger(__name__)


class PositionSizer:
    """
    Advanced position sizing with multiple algorithms and error handling.
    
    This class implements various position sizing methods including:
    - Fixed fractional sizing
    - Kelly Criterion
    - Volatility-based sizing
    - Optimal f
    - Risk parity
    - Machine learning-based sizing
    - Regime-adjusted sizing
    
    Attributes:
        risk_manager: Risk management instance
        params: Position sizing parameters
        sizing_history: Historical sizing decisions
    """
    
    def __init__(self, risk_manager) -> None:
        """
        Initialize the Position Sizer with error handling.
        
        Args:
            risk_manager: Risk management instance
        """
        try:
            # Validate risk manager
            if not risk_manager:
                raise ValueError("Risk manager is required")
                
            self.risk_manager = risk_manager
            
            # Sizing parameters with validation
            self.params: Dict[str, Any] = {
                'kelly_fraction': 0.25,  # Use 25% of Kelly for safety
                'max_position_pct': 0.1,  # Max 10% per position
                'min_position_size': 100,  # Minimum $100 position
                'confidence_threshold': 0.6,  # Minimum confidence for full size
                'volatility_lookback': 20,  # Days for volatility calculation
                'correlation_penalty': 0.3,  # Reduce size by 30% for correlated positions
                'regime_adjustments': {
                    'low_volatility': 1.2,
                    'normal': 1.0,
                    'high_volatility': 0.6,
                    'extreme_volatility': 0.3,
                    'trending': 1.1,
                    'ranging': 0.8
                }
            }
            
            # Validate parameters
            self._validate_parameters()
            
            # Initialize modules
            self.base_methods = BaseSizingMethods(self.params)
            self.advanced_methods = AdvancedSizingMethods(self.params)
            self.adjustments = SizeAdjustments(self.params)
            
            # Track sizing history
            self.sizing_history: List[Dict] = []
            
            # Error tracking
            self.error_count = 0
            self.max_errors = 50
            
            logger.info("PositionSizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PositionSizer: {e}")
            raise
    
    def _validate_parameters(self):
        """Validate sizing parameters"""
        for key, value in self.params.items():
            if key == 'regime_adjustments':
                continue
            if not isinstance(value, (int, float)) or value < 0:
                logger.warning(f"Invalid parameter {key}: {value}, using default")
                self.params[key] = 0.1 if 'pct' in key else 100
        
    def calculate_position_size(self, signal: Dict[str, Any], 
                              market_data: Dict[str, float], 
                              portfolio_state: Dict[str, Any]) -> float:
        """
        Calculate optimal position size using multiple methods with error handling.
        
        This method combines various sizing algorithms weighted by market
        conditions to determine the optimal position size.
        
        Args:
            signal: Trading signal with entry price, stop loss, confidence
            market_data: Current market conditions
            portfolio_state: Current portfolio positions and metrics
            
        Returns:
            Final position size in base currency
        """
        try:
            # Validate inputs
            if not self._validate_inputs(signal, market_data):
                return self.params['min_position_size']
            
            # Get base inputs
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', entry_price * 0.98)
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # Get available capital
            available_capital = self.risk_manager.get_available_capital()
            if available_capital <= 0:
                logger.warning("No available capital for position sizing")
                return 0
            
            # Calculate different position sizes
            sizes = self._calculate_all_sizes(signal, market_data, portfolio_state, 
                                            entry_price, stop_loss, available_capital)
            
            # Weight different methods based on market conditions
            weights = self.adjustments.calculate_method_weights(market_data, portfolio_state)
            
            # Calculate weighted position size
            weighted_size = self._calculate_weighted_size(sizes, weights)
            
            # Apply adjustments
            adjusted_size = self._apply_all_adjustments(
                weighted_size, signal, market_data, portfolio_state
            )
            
            # Apply limits
            final_size = self.adjustments.apply_limits(
                adjusted_size, entry_price, available_capital
            )
            
            # Log sizing decision
            self._log_sizing_decision(signal, sizes, weights, final_size)
            
            # Final validation
            if not isinstance(final_size, (int, float)) or final_size <= 0 or not np.isfinite(final_size):
                logger.error(f"Invalid final size: {final_size}")
                return self.params['min_position_size']
            
            return final_size
            
        except Exception as e:
            logger.error(f"Critical error calculating position size: {e}")
            logger.error(traceback.format_exc())
            self.error_count += 1
            
            # Return safe minimum
            return self.params['min_position_size']
    
    def _validate_inputs(self, signal: Dict[str, Any], 
                        market_data: Dict[str, float]) -> bool:
        """Validate input parameters"""
        # Validate signal
        if not signal or not isinstance(signal, dict):
            logger.error("Invalid signal format")
            return False
        
        # Validate prices
        entry_price = signal.get('entry_price', 0)
        if entry_price <= 0 or not np.isfinite(entry_price):
            logger.error(f"Invalid entry price: {entry_price}")
            return False
        
        stop_loss = signal.get('stop_loss', 0)
        if stop_loss <= 0 or not np.isfinite(stop_loss):
            logger.warning(f"Invalid stop loss: {stop_loss}, will use default")
        
        return True
    
    def _calculate_all_sizes(self, signal: Dict[str, Any], 
                           market_data: Dict[str, float],
                           portfolio_state: Dict[str, Any],
                           entry_price: float, stop_loss: float,
                           available_capital: float) -> Dict[str, float]:
        """Calculate sizes using all methods"""
        sizes = {}
        
        # Fixed fractional
        try:
            risk_per_trade = self.risk_manager.risk_params.get('risk_per_trade', 0.02)
            sizes['fixed_fractional'] = self.base_methods.fixed_fractional_size(
                entry_price, stop_loss, available_capital, risk_per_trade
            )
        except Exception as e:
            logger.error(f"Error in fixed fractional sizing: {e}")
            sizes['fixed_fractional'] = self.params['min_position_size']
        
        # Kelly criterion
        try:
            sizes['kelly'] = self.base_methods.kelly_criterion_size(
                signal, available_capital
            )
        except Exception as e:
            logger.error(f"Error in Kelly sizing: {e}")
            sizes['kelly'] = sizes.get('fixed_fractional', self.params['min_position_size'])
        
        # Volatility-based
        try:
            volatility = market_data.get('volatility', 0.02)
            sizes['volatility_based'] = self.base_methods.volatility_based_size(
                volatility, available_capital
            )
        except Exception as e:
            logger.error(f"Error in volatility sizing: {e}")
            sizes['volatility_based'] = sizes.get('fixed_fractional', self.params['min_position_size'])
        
        # Optimal f
        try:
            historical_results = self.advanced_methods.get_historical_results(signal)
            sizes['optimal_f'] = self.advanced_methods.optimal_f_size(
                historical_results, available_capital
            )
        except Exception as e:
            logger.error(f"Error in optimal f sizing: {e}")
            sizes['optimal_f'] = sizes.get('fixed_fractional', self.params['min_position_size'])
        
        # Risk parity
        try:
            positions = portfolio_state.get('positions', {})
            volatility = market_data.get('volatility', 0.02)
            sizes['risk_parity'] = self.advanced_methods.risk_parity_size(
                signal.get('symbol', 'UNKNOWN'), positions, volatility
            )
        except Exception as e:
            logger.error(f"Error in risk parity sizing: {e}")
            sizes['risk_parity'] = sizes.get('fixed_fractional', self.params['min_position_size'])
        
        # Machine learning
        try:
            features = self._extract_ml_features(signal, market_data, portfolio_state)
            sizes['machine_learning'] = self.advanced_methods.ml_based_size(
                features, available_capital
            )
        except Exception as e:
            logger.error(f"Error in ML sizing: {e}")
            sizes['machine_learning'] = sizes.get('fixed_fractional', self.params['min_position_size'])
        
        # Regime-adjusted
        try:
            regime = market_data.get('regime', 'normal')
            base_size = sizes.get('fixed_fractional', self.params['min_position_size'])
            sizes['regime_adjusted'] = self.advanced_methods.regime_adjusted_size(
                regime, base_size, market_data.get('regime_confidence', 0.7)
            )
        except Exception as e:
            logger.error(f"Error in regime sizing: {e}")
            sizes['regime_adjusted'] = sizes.get('fixed_fractional', self.params['min_position_size'])
        
        return sizes
    
    def _extract_ml_features(self, signal: Dict[str, Any], 
                           market_data: Dict[str, float],
                           portfolio_state: Dict[str, Any]) -> List[float]:
        """Extract features for ML-based sizing"""
        features = []
        
        # Safe feature extraction
        features.append(signal.get('confidence', 0.5))
        features.append(market_data.get('volatility', 0.02))
        features.append(market_data.get('volume_ratio', 1.0))
        features.append(len(portfolio_state.get('positions', {})))
        features.append(min(self.risk_manager.current_drawdown, 1.0))
        features.append(market_data.get('trend_strength', 0.0))
        features.append(signal.get('expected_return', 0.02))
        
        return features
    
    def _calculate_weighted_size(self, sizes: Dict[str, float], 
                               weights: Dict[str, float]) -> float:
        """Calculate weighted position size"""
        weighted_size = 0
        
        for method, weight in weights.items():
            if method in sizes:
                size = sizes[method]
                # Validate size
                if isinstance(size, (int, float)) and size > 0 and np.isfinite(size):
                    weighted_size += size * weight
                else:
                    logger.warning(f"Invalid size for {method}: {size}")
        
        # Ensure we have a valid size
        if weighted_size <= 0:
            logger.warning("Weighted size is zero or negative, using minimum")
            weighted_size = self.params['min_position_size']
            
        return weighted_size
    
    def _apply_all_adjustments(self, base_size: float, signal: Dict[str, Any],
                             market_data: Dict[str, float],
                             portfolio_state: Dict[str, Any]) -> float:
        """Apply all adjustments to the base size"""
        # Calculate correlation adjustment
        symbol = signal.get('symbol', 'UNKNOWN')
        positions = portfolio_state.get('positions', {})
        
        correlation_factor = self.adjustments.calculate_correlation_adjustment(
            symbol, positions, self.risk_manager._get_pair_correlation
        )
        
        # Apply adjustments
        adjusted_size = self.adjustments.apply_adjustments(
            base_size, signal, market_data, 
            self.risk_manager.current_drawdown, correlation_factor
        )
        
        return adjusted_size
    
    def _log_sizing_decision(self, signal: Dict[str, Any], 
                           sizes: Dict[str, float], 
                           weights: Dict[str, float], 
                           final_size: float) -> None:
        """Log position sizing decision for analysis"""
        try:
            decision = {
                'timestamp': pd.Timestamp.now(),
                'symbol': signal.get('symbol', 'UNKNOWN'),
                'signal_confidence': signal.get('confidence', 0.5),
                'sizes': sizes,
                'weights': weights,
                'final_size': final_size,
                'available_capital': self.risk_manager.get_available_capital()
            }
            
            self.sizing_history.append(decision)
            
            # Limit history size
            if len(self.sizing_history) > 1000:
                self.sizing_history = self.sizing_history[-1000:]
            
            logger.info(f"Position sizing for {signal.get('symbol', 'UNKNOWN')}: ${final_size:.2f}")
            logger.debug(f"Sizing details: {sizes}")
            
        except Exception as e:
            logger.error(f"Error logging sizing decision: {e}")
    
    def get_sizing_analytics(self) -> Dict[str, Any]:
        """Get analytics on sizing decisions"""
        if not self.sizing_history:
            return {}
            
        try:
            df = pd.DataFrame(self.sizing_history)
            
            analytics = {
                'total_decisions': len(self.sizing_history),
                'avg_size': df['final_size'].mean(),
                'std_size': df['final_size'].std(),
                'avg_confidence': df['signal_confidence'].mean(),
                'method_usage': self._analyze_method_usage(df),
                'size_distribution': self._analyze_size_distribution(df),
                'error_rate': self.error_count / max(len(self.sizing_history), 1)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error calculating sizing analytics: {e}")
            return {}
    
    def _analyze_method_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze which sizing methods contribute most"""
        try:
            # This would require storing method contributions
            # For now, return placeholder
            return {
                'fixed_fractional': 0.2,
                'kelly': 0.15,
                'volatility_based': 0.15,
                'optimal_f': 0.1,
                'risk_parity': 0.1,
                'machine_learning': 0.15,
                'regime_adjusted': 0.15
            }
        except Exception as e:
            logger.error(f"Error analyzing method usage: {e}")
            return {}
    
    def _analyze_size_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze size distribution"""
        try:
            return {
                'min': df['final_size'].min(),
                'max': df['final_size'].max(),
                'p25': df['final_size'].quantile(0.25),
                'p50': df['final_size'].quantile(0.50),
                'p75': df['final_size'].quantile(0.75)
            }
        except Exception as e:
            logger.error(f"Error analyzing size distribution: {e}")
            return {}

"""
REFACTORING SUMMARY:
- Original file: 1200+ lines
- Refactored position_sizer.py: ~450 lines
- Created 3 modular components:
  1. base_sizing.py - Basic sizing methods (fixed fractional, Kelly, volatility)
  2. advanced_sizing.py - Advanced methods (optimal f, risk parity, ML, regime)
  3. size_adjustments.py - Adjustments, limits, and weighting logic
- Benefits:
  * Clear separation of sizing methods
  * Easier to test individual algorithms
  * More maintainable code structure
  * Simplified main coordinator
  * Easy to add new sizing methods
"""
