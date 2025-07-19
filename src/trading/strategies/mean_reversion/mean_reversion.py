"""
File: mean_reversion.py
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
import logging
from ..risk_manager.risk_manager import RiskManager
from ...utils.logger import setup_logger

# Import modules
from .modules.indicators import MeanReversionIndicators
from .modules.signal_generator import SignalGenerator, MeanReversionSignal
from .modules.position_manager import PositionManager
from .modules.data_validator import DataValidator

logger = setup_logger(__name__)


class MeanReversionStrategy:
    """Statistical arbitrage and mean reversion strategy with comprehensive error handling"""
    
    def __init__(self, risk_manager: RiskManager):
        """
        Initialize mean reversion strategy with error handling.
        
        Args:
            risk_manager: Risk management instance
        """
        # Validate risk manager
        if not risk_manager:
            raise ValueError("Risk manager is required for MeanReversionStrategy")
        
        self.risk_manager = risk_manager
        
        # Strategy parameters with validation
        self.params = {
            'lookback_period': 20,
            'entry_z_score': 2.0,
            'exit_z_score': 0.5,
            'max_z_score': 3.0,
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'min_volatility': 0.01,
            'max_volatility': 0.10,
            'atr_multiplier_sl': 3.0,
            'position_hold_bars': 48,  # Maximum bars to hold position
            'min_confidence': 0.65
        }
        
        # Initialize modules
        self.data_validator = DataValidator()
        
        # Validate parameters
        if not self.data_validator.validate_parameters(self.params):
            raise ValueError("Invalid strategy parameters")
        
        # Initialize other modules with validated params
        self.indicator_calculator = MeanReversionIndicators(self.params)
        self.signal_generator = SignalGenerator(self.params, risk_manager)
        self.position_manager = PositionManager(self.params)
        
        # Error tracking
        self.error_stats = {
            'calculation_errors': 0,
            'signal_generation_errors': 0,
            'position_update_errors': 0,
            'last_error': None
        }
        
        logger.info("MeanReversionStrategy initialized successfully")
    
    def analyze(self, df: pd.DataFrame, ml_predictions: Optional[Dict] = None) -> List[MeanReversionSignal]:
        """
        Analyze market for mean reversion opportunities with comprehensive error handling.
        
        Args:
            df: Market data DataFrame
            ml_predictions: Optional ML predictions
            
        Returns:
            List of mean reversion signals
        """
        signals = []
        
        try:
            # Validate and clean data
            df_clean = self.data_validator.validate_and_clean(df)
            if df_clean is None:
                return signals
            
            # Ensure sufficient data
            min_periods = max(
                self.params['lookback_period'],
                self.params['bb_period'],
                self.params['rsi_period']
            )
            
            if len(df_clean) < min_periods:
                logger.warning(f"Insufficient data: {len(df_clean)} rows, need {min_periods}")
                return signals
            
            # Calculate mean reversion indicators
            indicators = self.indicator_calculator.calculate_all(df_clean)
            
            if not indicators:
                logger.error("Failed to calculate indicators")
                self.error_stats['calculation_errors'] += 1
                return signals
            
            # Generate signals
            signals = self.signal_generator.generate_signals(
                df_clean, indicators, ml_predictions
            )
            
        except Exception as e:
            logger.error(f"Critical error in mean reversion analysis: {e}")
            self.error_stats['calculation_errors'] += 1
            self.error_stats['last_error'] = str(e)
        
        return signals
    
    def update_positions(self, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Update positions based on current market data with error handling"""
        try:
            actions = self.position_manager.update_positions(
                current_data, self.indicator_calculator
            )
            return actions
        except Exception as e:
            logger.error(f"Error in update_positions: {e}")
            self.error_stats['position_update_errors'] += 1
            return []
    
    def add_position(self, symbol: str, signal: MeanReversionSignal):
        """Add a new position based on signal"""
        position_data = {
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'direction': signal.direction,
            'entry_z_score': signal.z_score
        }
        self.position_manager.add_position(symbol, position_data)
    
    def has_position(self, symbol: str) -> bool:
        """Check if strategy has position for symbol"""
        return self.position_manager.has_position(symbol)
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        position_stats = self.position_manager.get_statistics()
        validation_stats = self.data_validator.get_validation_stats()
        
        # Aggregate error stats from all modules
        total_errors = (
            self.error_stats['calculation_errors'] +
            self.signal_generator.error_stats['signal_generation_errors'] +
            self.position_manager.error_stats['position_update_errors']
        )
        
        return {
            'name': 'Mean Reversion',
            'stats': position_stats['stats'],
            'active_positions': position_stats['active_positions'],
            'parameters': self.params,
            'error_stats': {
                'calculation_errors': self.error_stats['calculation_errors'],
                'signal_generation_errors': self.signal_generator.error_stats['signal_generation_errors'],
                'position_update_errors': self.position_manager.error_stats['position_update_errors'],
                'total_errors': total_errors,
                'last_error': self.error_stats['last_error']
            },
            'validation_stats': validation_stats
        }
    
    @property
    def active_positions(self):
        """Get active positions (for backward compatibility)"""
        return self.position_manager.active_positions
    
    @property
    def stats(self):
        """Get strategy stats (for backward compatibility)"""
        return self.position_manager.stats
