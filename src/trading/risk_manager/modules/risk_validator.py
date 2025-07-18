"""
Risk validation module for trading risk management.
Handles pre-trade risk checks and validation logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import traceback

from utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskValidator:
    """Validates trades against risk parameters"""
    
    def __init__(self, risk_params: Dict[str, float]):
        self.risk_params = risk_params
        self.risk_breaches: List[Dict[str, Any]] = []
        self.emergency_mode = False
        self.emergency_triggered_at = None
        
    def validate_risk_params(self) -> bool:
        """Validate risk parameters are within acceptable ranges"""
        try:
            # Validate ranges
            validations = [
                ('max_position_size', 0, 1),
                ('max_total_exposure', 0, 2),
                ('max_correlation', 0, 1),
                ('stop_loss_pct', 0, 0.5),
                ('max_daily_loss', 0, 0.5),
                ('max_drawdown', 0, 0.5),
                ('risk_per_trade', 0, 0.1),
                ('var_confidence', 0.5, 0.99),
                ('liquidity_buffer', 0, 0.5),
                ('max_leverage', 1, 10)
            ]
            
            for param, min_val, max_val in validations:
                if param in self.risk_params:
                    value = self.risk_params[param]
                    if not (min_val <= value <= max_val):
                        logger.warning(f"Risk parameter {param}={value} out of range [{min_val}, {max_val}]")
                        self.risk_params[param] = np.clip(value, min_val, max_val)
            
            # Validate integers
            self.risk_params['position_limit'] = max(1, int(self.risk_params.get('position_limit', 10)))
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating risk parameters: {e}")
            # Use conservative defaults on error
            self.risk_params['max_position_size'] = 0.05
            self.risk_params['max_total_exposure'] = 0.5
            return False
    
    def check_pre_trade_risk(self, symbol: str, side: str, size: float, price: float,
                           stop_loss: Optional[float], current_capital: float,
                           position_count: int, total_exposure: float,
                           daily_loss: float = 0) -> Tuple[bool, str]:
        """
        Check if trade passes risk criteria with comprehensive error handling.
        
        Args:
            symbol: Trading symbol
            side: Trade side (long/short)
            size: Position size
            price: Entry price
            stop_loss: Optional stop loss price
            current_capital: Current capital
            position_count: Current number of positions
            total_exposure: Current total exposure
            daily_loss: Current daily loss
            
        Returns:
            Tuple of (passed, reason) where passed is boolean and reason is string
        """
        try:
            # Emergency mode check
            if self.emergency_mode:
                logger.warning("Risk manager in emergency mode")
                return False, "Emergency mode active"
            
            # Check daily loss limit
            if daily_loss > 0:  # daily_loss should be negative for losses
                daily_loss_pct = abs(daily_loss) / current_capital
                if daily_loss_pct >= self.risk_params['max_daily_loss']:
                    self.log_risk_breach("daily_loss_limit", 
                                       f"Daily loss limit of {self.risk_params['max_daily_loss']:.1%} reached")
                    return False, "Daily loss limit reached"
            
            # Check position limit
            if position_count >= self.risk_params['position_limit']:
                return False, f"Position limit ({self.risk_params['position_limit']}) reached"
            
            # Check position size
            position_value = abs(size * price)
            max_position_value = current_capital * self.risk_params['max_position_size']
            
            if position_value > max_position_value:
                return False, f"Position size (${position_value:,.2f}) exceeds limit (${max_position_value:,.2f})"
            
            # Check minimum position size
            min_position_value = current_capital * self.risk_params.get('min_position_size', 0.001)
            if position_value < min_position_value:
                return False, f"Position size below minimum (${min_position_value:,.2f})"
            
            # Check total exposure
            new_exposure = total_exposure + position_value
            max_exposure = current_capital * self.risk_params['max_total_exposure']
            
            if new_exposure > max_exposure:
                return False, f"Total exposure (${new_exposure:,.2f}) would exceed limit (${max_exposure:,.2f})"
            
            # Check liquidity
            required_liquidity = current_capital * self.risk_params['liquidity_buffer']
            available_liquidity = current_capital - total_exposure
            
            if available_liquidity - position_value < required_liquidity:
                return False, f"Insufficient liquidity buffer (need ${required_liquidity:,.2f})"
            
            # Check stop loss risk
            if stop_loss:
                risk_amount = abs(price - stop_loss) * size
                max_risk = current_capital * self.risk_params['risk_per_trade']
                
                if risk_amount > max_risk:
                    return False, f"Stop loss risk (${risk_amount:,.2f}) exceeds limit (${max_risk:,.2f})"
            
            # Check leverage
            total_position_value = new_exposure
            leverage = total_position_value / current_capital if current_capital > 0 else 0
            
            if leverage > self.risk_params['max_leverage']:
                return False, f"Leverage ({leverage:.2f}x) exceeds limit ({self.risk_params['max_leverage']}x)"
            
            return True, "Risk check passed"
            
        except Exception as e:
            logger.error(f"Error in pre-trade risk check: {e}")
            logger.error(traceback.format_exc())
            return False, f"Risk calculation error: {str(e)}"
    
    def check_correlation_risk(self, symbol: str, existing_positions: Dict[str, Any],
                              correlation_matrix: pd.DataFrame = None) -> bool:
        """Check if adding position increases correlation risk"""
        try:
            if len(existing_positions) == 0:
                return True
            
            # Get correlation with existing positions
            high_correlation_count = 0
            
            for existing_symbol in existing_positions:
                correlation = self._get_pair_correlation(symbol, existing_symbol, correlation_matrix)
                
                # Validate correlation
                if not -1 <= correlation <= 1:
                    logger.warning(f"Invalid correlation: {correlation}")
                    correlation = 0.5
                
                if correlation > self.risk_params['max_correlation']:
                    high_correlation_count += 1
            
            # Allow if less than half of positions are highly correlated
            return high_correlation_count < len(existing_positions) / 2
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return False
    
    def _get_pair_correlation(self, symbol1: str, symbol2: str, 
                            correlation_matrix: pd.DataFrame = None) -> float:
        """Get correlation between two symbols with error handling"""
        try:
            if symbol1 == symbol2:
                return 1.0
            
            # Check if we have correlation data
            if correlation_matrix is not None and not correlation_matrix.empty:
                if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[symbol1, symbol2]
                    if np.isfinite(corr):
                        return float(corr)
            
            # Default correlation for crypto pairs
            # In production, this would use actual historical correlation
            if 'BTC' in symbol1 and 'BTC' in symbol2:
                return 0.9
            elif any(coin in symbol1 and coin in symbol2 for coin in ['ETH', 'BTC']):
                return 0.7
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error getting pair correlation: {e}")
            return 0.5
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              current_capital: float, symbol: str = None,
                              current_drawdown: float = 0,
                              existing_positions: Dict = None) -> float:
        """
        Calculate position size based on risk parameters with error handling.
        
        Args:
            entry_price: Entry price for position
            stop_loss: Stop loss price
            current_capital: Current capital
            symbol: Trading symbol
            current_drawdown: Current drawdown percentage
            existing_positions: Existing positions for correlation adjustment
            
        Returns:
            Calculated position size
        """
        try:
            # Validate inputs
            if entry_price <= 0 or not np.isfinite(entry_price):
                logger.error(f"Invalid entry price: {entry_price}")
                return 0
            
            if stop_loss <= 0 or not np.isfinite(stop_loss):
                logger.error(f"Invalid stop loss: {stop_loss}")
                return 0
            
            # Calculate price risk
            price_risk = abs(entry_price - stop_loss)
            
            if price_risk == 0:
                logger.warning("Zero price risk, cannot calculate position size")
                return 0
            
            # Risk amount
            risk_amount = current_capital * self.risk_params['risk_per_trade']
            
            # Base position size
            position_size = risk_amount / price_risk
            
            # Apply position size limit
            max_position_value = current_capital * self.risk_params['max_position_size']
            max_position_size = max_position_value / entry_price
            
            position_size = min(position_size, max_position_size)
            
            # Apply adjustments
            correlation_factor = self._get_correlation_adjustment(symbol, existing_positions)
            drawdown_factor = self._get_drawdown_adjustment(current_drawdown)
            volatility_factor = self._get_volatility_adjustment(symbol)
            
            # Validate adjustment factors
            for factor in [correlation_factor, drawdown_factor, volatility_factor]:
                if not 0 < factor <= 1:
                    logger.warning(f"Invalid adjustment factor: {factor}")
                    return 0
            
            position_size *= correlation_factor * drawdown_factor * volatility_factor
            
            # Final validation
            if not np.isfinite(position_size) or position_size < 0:
                logger.error(f"Invalid calculated position size: {position_size}")
                return 0
            
            # Apply minimum position size
            min_size = self.risk_params.get('min_position_size', 0.001) * current_capital / entry_price
            if position_size < min_size:
                logger.info(f"Position size below minimum: {position_size} < {min_size}")
                return 0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _get_correlation_adjustment(self, symbol: str, existing_positions: Dict = None) -> float:
        """Get position size adjustment based on correlation"""
        try:
            if not existing_positions or len(existing_positions) == 0:
                return 1.0
            
            # Calculate average correlation with existing positions
            correlations = []
            for existing_symbol in existing_positions:
                corr = self._get_pair_correlation(symbol, existing_symbol)
                correlations.append(abs(corr))
            
            avg_correlation = np.mean(correlations)
            
            # Reduce size for high correlation
            if avg_correlation > 0.7:
                return 0.5
            elif avg_correlation > 0.5:
                return 0.75
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return 0.5  # Conservative default
    
    def _get_drawdown_adjustment(self, current_drawdown: float) -> float:
        """Adjust position size based on current drawdown"""
        try:
            if current_drawdown < 0.05:
                return 1.0
            elif current_drawdown < 0.10:
                return 0.75
            elif current_drawdown < 0.15:
                return 0.5
            else:
                return 0.25
                
        except Exception as e:
            logger.error(f"Error calculating drawdown adjustment: {e}")
            return 0.25  # Conservative default
    
    def _get_volatility_adjustment(self, symbol: str) -> float:
        """Adjust position size based on volatility"""
        try:
            # In production, would use actual volatility data
            # For now, return a conservative adjustment
            return 0.8
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 0.5
    
    def get_risk_adjusted_leverage(self, current_drawdown: float, position_count: int) -> float:
        """
        Calculate appropriate leverage based on current risk.
        
        Returns:
            Recommended leverage multiplier
        """
        try:
            base_leverage = min(self.risk_params['max_leverage'], 3.0)
            
            # Adjust for drawdown
            if current_drawdown > 0.1:
                base_leverage *= 0.5
            elif current_drawdown > 0.05:
                base_leverage *= 0.75
            
            # Adjust for volatility (placeholder)
            volatility_adjustment = 0.8
            
            # Adjust for correlation
            if position_count > 5:
                correlation_adjustment = 0.7
            else:
                correlation_adjustment = 1.0
            
            # Emergency mode
            if self.emergency_mode:
                return 1.0
            
            final_leverage = base_leverage * volatility_adjustment * correlation_adjustment
            
            return max(1.0, min(self.risk_params['max_leverage'], final_leverage))
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted leverage: {e}")
            return 1.0
    
    def enter_emergency_mode(self, reason: str):
        """Enter emergency mode to protect remaining capital"""
        logger.critical(f"Entering emergency mode: {reason}")
        self.emergency_mode = True
        self.emergency_triggered_at = pd.Timestamp.now()
        
        # Log risk breach
        self.log_risk_breach("emergency_mode", reason)
    
    def exit_emergency_mode(self):
        """Exit emergency mode"""
        logger.info("Exiting emergency mode")
        self.emergency_mode = False
        self.emergency_triggered_at = None
    
    def log_risk_breach(self, breach_type: str, details: str) -> None:
        """
        Log risk limit breach with error handling.
        
        Args:
            breach_type: Type of breach
            details: Breach details
        """
        try:
            breach = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'type': breach_type,
                'details': details
            }
            
            self.risk_breaches.append(breach)
            
            # Keep only last 100 breaches
            if len(self.risk_breaches) > 100:
                self.risk_breaches = self.risk_breaches[-100:]
            
            logger.warning(f"Risk breach: {breach_type} - {details}")
            
        except Exception as e:
            logger.error(f"Error logging risk breach: {e}")
