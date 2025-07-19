"""
File: momentum.py
Modified: 2024-12-19
Changes Summary:
- Refactored into modular architecture
- Separated concerns: indicators, signals, positions, validation
- Maintained all original functionality and error handling
- Performance impact: minimal (same as original)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import traceback
from ..risk_manager.risk_manager import RiskManager
from ...utils.logger import setup_logger

# Import modules
from .modules.indicators import MomentumIndicators
from .modules.signal_generator import SignalGenerator, MomentumSignal
from .modules.position_manager import PositionManager
from .modules.data_validator import DataValidator

logger = setup_logger(__name__)


class MomentumStrategy:
    """Advanced momentum trading strategy with comprehensive error handling"""
    
    def __init__(self, risk_manager: RiskManager):
        """
        Initialize momentum strategy with error handling.
        
        Args:
            risk_manager: Risk management instance
        """
        # Validate risk manager
        if not risk_manager:
            raise ValueError("Risk manager is required for MomentumStrategy")
        
        self.risk_manager = risk_manager
        
        # Strategy parameters with validation
        self.params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'adx_threshold': 25,
            'volume_multiplier': 1.5,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'min_confidence': 0.6,
            'min_data_points': 50,  # Minimum data required
            'max_positions': 5
        }
        
        # Initialize modules
        self.data_validator = DataValidator(self.params)
        
        # Validate parameters
        if not self.data_validator.validate_parameters(self.params):
            raise ValueError("Invalid momentum strategy parameters")
        
        # Initialize other modules with validated params
        self.indicator_calculator = MomentumIndicators(self.params)
        self.signal_generator = SignalGenerator(self.params, risk_manager)
        self.position_manager = PositionManager(self.params)
        
        # Performance tracking
        self.errors = 0
        
        logger.info("MomentumStrategy initialized successfully")
    
    def analyze(self, df: pd.DataFrame, ml_predictions: Optional[Dict] = None) -> List[MomentumSignal]:
        """
        Analyze market data for momentum signals with comprehensive error handling.
        
        Args:
            df: Market data DataFrame
            ml_predictions: Optional ML predictions
            
        Returns:
            List of momentum signals
        """
        signals = []
        
        try:
            # Validate input data
            if not self.data_validator.validate_market_data(df):
                logger.error("Invalid market data provided")
                return []
            
            # Check data quality
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
            if self.data_validator.has_quality_issues(symbol):
                logger.warning(f"Skipping {symbol} due to data quality issues")
                return []
            
            # Check cache
            cache_key = f"{symbol}_{pd.Timestamp.now().floor('T')}"  # Per minute cache
            if cache_key in self.indicator_calculator.calculation_cache:
                indicators = self.indicator_calculator.calculation_cache[cache_key]
            else:
                # Calculate indicators with error handling
                indicators = self.indicator_calculator.calculate_all(df)
                if indicators is None:
                    self.data_validator.record_quality_issue(symbol)
                    return []
                
                self.indicator_calculator.calculation_cache[cache_key] = indicators
                # Clean old cache entries
                self.indicator_calculator.clean_cache()
            
            # Generate signals
            signals = self.signal_generator.generate_signals(
                df, indicators, ml_predictions
            )
            
            # Limit signals based on position count
            if len(self.position_manager.active_positions) >= self.params['max_positions']:
                logger.info("Maximum positions reached, filtering signals")
                available_slots = self.params['max_positions'] - len(self.position_manager.active_positions)
                signals = self.signal_generator.filter_best_signals(signals, max(1, available_slots))
            
            return signals
            
        except Exception as e:
            logger.error(f"Critical error in momentum analysis: {e}")
            logger.error(traceback.format_exc())
            self.errors += 1
            return []
    
    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Update positions based on current prices with error handling"""
        try:
            actions = self.position_manager.update_positions(current_prices)
            return actions
        except Exception as e:
            logger.error(f"Error in update_positions: {e}")
            self.errors += 1
            return []
    
    def add_position(self, symbol: str, signal: MomentumSignal):
        """Add a new position based on signal"""
        position_data = {
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'direction': signal.direction,
            'strength': signal.strength,
            'confidence': signal.confidence
        }
        self.position_manager.add_position(symbol, position_data)
    
    def has_position(self, symbol: str) -> bool:
        """Check if strategy has position for symbol"""
        return self.position_manager.has_position(symbol)
    
    def get_strategy_metrics(self) -> Dict:
        """Get comprehensive strategy metrics with error handling"""
        try:
            # Get performance metrics from position manager
            performance_metrics = self.position_manager.get_performance_metrics()
            
            # Get validation stats
            validation_stats = self.data_validator.get_validation_stats()
            
            # Aggregate all metrics
            return {
                'name': 'Momentum',
                'performance': performance_metrics,
                'active_positions': len(self.position_manager.active_positions),
                'parameters': self.params.copy(),
                'health': {
                    'calculation_failures': self.indicator_calculator.get_calculation_failures(),
                    'data_quality_issues': validation_stats,
                    'cache_size': len(self.indicator_calculator.calculation_cache),
                    'total_errors': self.errors + self.signal_generator.errors + self.position_manager.errors
                }
            }
        except Exception as e:
            logger.error(f"Error getting strategy metrics: {e}")
            return {
                'name': 'Momentum',
                'error': str(e)
            }
    
    @property
    def active_positions(self):
        """Get active positions (for backward compatibility)"""
        return self.position_manager.active_positions
    
    @property
    def performance(self):
        """Get performance metrics (for backward compatibility)"""
        return self.position_manager.performance
    
    @property
    def calculation_cache(self):
        """Get calculation cache (for backward compatibility)"""
        return self.indicator_calculator.calculation_cache
    
    @property
    def data_quality_issues(self):
        """Get data quality issues (for backward compatibility)"""
        return self.data_validator.data_quality_issues


# Re-export for backward compatibility
__all__ = ['MomentumStrategy', 'MomentumSignal']

"""
MODULARIZATION_SUMMARY:
- Total modules created: 4
- Original file size: 39,632 bytes (~1,150 lines)
- Main file size after refactoring: ~7KB (~200 lines)
- Module breakdown:
  1. indicators.py (~350 lines) - All technical indicator calculations
  2. signal_generator.py (~400 lines) - Signal generation and momentum checking
  3. position_manager.py (~300 lines) - Position tracking and exit management
  4. data_validator.py (~200 lines) - Data validation and quality checking
- Maintained 100% backward compatibility
- All error handling preserved and enhanced
- Performance impact: None (same computational complexity)
"""
