"""
Base sizing methods module for position sizing.
Implements fundamental position sizing algorithms.
"""

import numpy as np
from typing import Dict, Any, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseSizingMethods:
    """Implements basic position sizing methods"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
    def fixed_fractional_size(self, entry_price: float, stop_loss: float, 
                             available_capital: float, risk_per_trade: float) -> float:
        """
        Fixed fractional position sizing with error handling.
        
        Args:
            entry_price: Entry price for position
            stop_loss: Stop loss price
            available_capital: Available capital for trading
            risk_per_trade: Risk per trade as fraction
            
        Returns:
            Position size in base currency
        """
        try:
            if available_capital <= 0:
                logger.warning("No available capital")
                return 0
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit <= 0 or not np.isfinite(risk_per_unit):
                logger.warning(f"Invalid risk per unit: {risk_per_unit}")
                return self.params['min_position_size']
            
            # Position size in units
            position_units = (available_capital * risk_per_trade) / risk_per_unit
            
            # Convert to dollar size
            position_size = position_units * entry_price
            
            # Validate result
            if not np.isfinite(position_size) or position_size < 0:
                return self.params['min_position_size']
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error in fixed fractional sizing: {e}")
            return self.params['min_position_size']
    
    def kelly_criterion_size(self, signal: Dict[str, Any], 
                           available_capital: float) -> float:
        """
        Kelly criterion position sizing with error handling.
        
        Kelly formula: f = (p*b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        
        Args:
            signal: Trading signal
            available_capital: Available capital
            
        Returns:
            Position size based on Kelly criterion
        """
        try:
            # Estimate win probability and win/loss ratio
            win_prob = signal.get('win_probability', 0.55)
            
            # Validate probability
            if not 0 < win_prob < 1:
                logger.warning(f"Invalid win probability: {win_prob}")
                win_prob = 0.55
            
            # Calculate expected win/loss amounts
            entry_price = signal.get('entry_price', 0)
            if entry_price <= 0:
                return self.params['min_position_size']
                
            take_profit = signal.get('take_profit', entry_price * 1.03)
            stop_loss = signal.get('stop_loss', entry_price * 0.98)
            
            win_amount = abs(take_profit - entry_price) / entry_price
            loss_amount = abs(entry_price - stop_loss) / entry_price
            
            # Validate amounts
            if loss_amount <= 0 or not np.isfinite(win_amount) or not np.isfinite(loss_amount):
                return self.params['min_position_size']
            
            # Kelly formula
            b = win_amount / loss_amount
            q = 1 - win_prob
            
            kelly_fraction = (win_prob * b - q) / b
            
            # Apply safety factor
            kelly_fraction *= self.params['kelly_fraction']
            
            # Ensure non-negative and reasonable
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Convert to position size
            if available_capital <= 0:
                return self.params['min_position_size']
                
            position_size = available_capital * kelly_fraction
            
            return max(position_size, self.params['min_position_size'])
            
        except Exception as e:
            logger.error(f"Error in Kelly criterion sizing: {e}")
            return self.params['min_position_size']
    
    def volatility_based_size(self, volatility: float, 
                            available_capital: float) -> float:
        """
        Volatility-based position sizing with error handling.
        
        Size positions to target consistent portfolio volatility.
        
        Args:
            volatility: Asset volatility
            available_capital: Available capital
            
        Returns:
            Volatility-adjusted position size
        """
        try:
            # Validate volatility
            if volatility <= 0 or volatility > 10 or not np.isfinite(volatility):
                logger.warning(f"Invalid volatility: {volatility}, using default")
                volatility = 0.02
            
            # Target volatility (e.g., 1% portfolio volatility)
            target_volatility = 0.01
            
            # Calculate position size to achieve target volatility
            if available_capital <= 0:
                return self.params['min_position_size']
            
            volatility_ratio = target_volatility / volatility
            # Cap volatility ratio
            volatility_ratio = min(volatility_ratio, 2.0)  # Max 2x leverage for low vol
            
            position_size = available_capital * volatility_ratio * 0.1  # 10% base allocation
            
            return max(position_size, self.params['min_position_size'])
            
        except Exception as e:
            logger.error(f"Error in volatility-based sizing: {e}")
            return self.params['min_position_size']
