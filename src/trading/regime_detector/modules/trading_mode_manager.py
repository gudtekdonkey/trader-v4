"""
Trading Mode Manager Module

Manages trading mode recommendations based on detected market regimes.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TradingModeManager:
    """
    Manages trading parameters and recommendations based on market regime.
    
    This class provides regime-specific trading parameters including
    position sizing, risk limits, and preferred strategies.
    """
    
    def __init__(self):
        """Initialize the trading mode manager."""
        self.trading_modes = self._initialize_trading_modes()
        self.hedge_thresholds = self._initialize_hedge_thresholds()
    
    def _initialize_trading_modes(self) -> Dict[int, Dict[str, Any]]:
        """Initialize trading mode configurations for each regime."""
        return {
            0: {  # Low volatility
                'name': 'Range Trading',
                'position_size_multiplier': 1.5,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.015,
                'preferred_strategies': ['mean_reversion', 'market_making'],
                'max_leverage': 5,
                'risk_allocation': 0.08,  # 8% portfolio risk
                'holding_period': 'short',  # minutes to hours
                'entry_threshold': 0.7,  # Lower threshold for entries
                'exit_threshold': 0.3,  # Quick exits
                'pyramiding_allowed': True,
                'max_positions': 5
            },
            1: {  # Normal
                'name': 'Trend Following',
                'position_size_multiplier': 1.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.03,
                'preferred_strategies': ['momentum', 'trend_following'],
                'max_leverage': 3,
                'risk_allocation': 0.06,  # 6% portfolio risk
                'holding_period': 'medium',  # hours to days
                'entry_threshold': 0.75,
                'exit_threshold': 0.25,
                'pyramiding_allowed': True,
                'max_positions': 3
            },
            2: {  # High volatility
                'name': 'Reduced Risk',
                'position_size_multiplier': 0.5,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.05,
                'preferred_strategies': ['momentum', 'breakout'],
                'max_leverage': 2,
                'risk_allocation': 0.04,  # 4% portfolio risk
                'holding_period': 'medium',
                'entry_threshold': 0.8,  # Higher threshold for entries
                'exit_threshold': 0.2,
                'pyramiding_allowed': False,
                'max_positions': 2
            },
            3: {  # Extreme volatility
                'name': 'Risk Off',
                'position_size_multiplier': 0.25,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.08,
                'preferred_strategies': ['arbitrage', 'market_neutral'],
                'max_leverage': 1,
                'risk_allocation': 0.02,  # 2% portfolio risk
                'holding_period': 'short',
                'entry_threshold': 0.9,  # Very high threshold
                'exit_threshold': 0.1,  # Quick exits
                'pyramiding_allowed': False,
                'max_positions': 1
            }
        }
    
    def _initialize_hedge_thresholds(self) -> Dict[str, float]:
        """Initialize hedging thresholds."""
        return {
            'portfolio_beta_threshold': 1.2,
            'correlation_threshold': 0.8,
            'volatility_threshold': 0.25,
            'drawdown_threshold': 0.08,
            'var_threshold': 0.05
        }
    
    def get_trading_mode(self, regime: int) -> Dict[str, Any]:
        """
        Get trading parameters for a specific regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary with trading parameters
        """
        if regime in self.trading_modes:
            return self.trading_modes[regime].copy()
        else:
            logger.warning(f"Unknown regime {regime}, using default (Normal)")
            return self.trading_modes.get(1, self._get_default_mode())
    
    def _get_default_mode(self) -> Dict[str, Any]:
        """Get default trading mode for error cases."""
        return {
            'name': 'Default Safe Mode',
            'position_size_multiplier': 0.5,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.03,
            'preferred_strategies': ['market_neutral'],
            'max_leverage': 1,
            'risk_allocation': 0.03,
            'holding_period': 'short',
            'entry_threshold': 0.85,
            'exit_threshold': 0.15,
            'pyramiding_allowed': False,
            'max_positions': 1
        }
    
    def get_position_size_adjustment(self, regime: int, base_size: float) -> float:
        """
        Calculate adjusted position size based on regime.
        
        Args:
            regime: Current market regime
            base_size: Base position size
            
        Returns:
            Adjusted position size
        """
        try:
            mode = self.get_trading_mode(regime)
            multiplier = mode.get('position_size_multiplier', 1.0)
            
            adjusted_size = base_size * multiplier
            
            # Apply safety limits
            max_size = base_size * 2.0  # Never more than 2x base
            min_size = base_size * 0.1  # Never less than 10% of base
            
            return max(min_size, min(adjusted_size, max_size))
            
        except Exception as e:
            logger.error(f"Error calculating position size adjustment: {e}")
            return base_size * 0.5  # Conservative fallback
    
    def should_enter_position(self, regime: int, signal_strength: float) -> bool:
        """
        Determine if a position should be entered based on regime and signal.
        
        Args:
            regime: Current market regime
            signal_strength: Strength of entry signal (0-1)
            
        Returns:
            True if position should be entered
        """
        try:
            mode = self.get_trading_mode(regime)
            threshold = mode.get('entry_threshold', 0.75)
            
            return signal_strength >= threshold
            
        except Exception as e:
            logger.error(f"Error checking entry condition: {e}")
            return False  # Conservative fallback
    
    def get_risk_limits(self, regime: int) -> Dict[str, float]:
        """
        Get risk limits for the current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary with risk limits
        """
        try:
            mode = self.get_trading_mode(regime)
            
            return {
                'stop_loss_pct': mode.get('stop_loss_pct', 0.02),
                'take_profit_pct': mode.get('take_profit_pct', 0.03),
                'max_leverage': mode.get('max_leverage', 1),
                'risk_allocation': mode.get('risk_allocation', 0.03),
                'max_positions': mode.get('max_positions', 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting risk limits: {e}")
            return {
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.03,
                'max_leverage': 1,
                'risk_allocation': 0.03,
                'max_positions': 1
            }
