"""
File: risk_manager.py
Modified: 2024-12-19
Refactored: 2025-07-18

Comprehensive risk management system with modular architecture.
This file coordinates the risk management modules for better maintainability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import traceback

# Import risk management modules
from .modules.risk_metrics import RiskMetrics, RiskCalculator
from .modules.position_manager import PositionManager
from .modules.risk_validator import RiskValidator

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskManager:
    """
    Comprehensive risk management system with error handling.
    
    This class handles all aspects of trading risk management including:
    - Position sizing and limits
    - Risk metrics calculation
    - Stop loss management
    - Drawdown monitoring
    - VaR and CVaR calculations
    - Portfolio exposure management
    
    Attributes:
        initial_capital: Starting capital
        current_capital: Current capital including P&L
        positions: Active positions (managed by PositionManager)
        trade_history: Historical trades (managed by PositionManager)
        risk_params: Risk management parameters
        daily_pnl: Daily P&L tracking
        equity_curve: Historical equity values
        high_water_mark: Highest portfolio value
        current_drawdown: Current drawdown from HWM
        max_drawdown: Maximum historical drawdown
    """
    
    def __init__(self, initial_capital: float = 100000) -> None:
        """
        Initialize the Risk Manager with validation.
        
        Args:
            initial_capital: Starting capital amount
        """
        # Validate initial capital
        if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
            logger.error(f"Invalid initial capital: {initial_capital}")
            raise ValueError("Initial capital must be a positive number")
        
        self.initial_capital = float(initial_capital)
        self.current_capital = self.initial_capital
        
        # Risk parameters with validation
        self.risk_params: Dict[str, float] = {
            'max_position_size': 0.1,  # 10% of capital per position
            'max_total_exposure': 0.8,  # 80% total exposure
            'max_correlation': 0.7,  # Maximum correlation between positions
            'stop_loss_pct': 0.02,  # 2% stop loss
            'max_daily_loss': 0.05,  # 5% daily loss limit
            'max_drawdown': 0.20,  # 20% maximum drawdown
            'position_limit': 10,  # Maximum number of positions
            'risk_per_trade': 0.02,  # 2% risk per trade
            'var_confidence': 0.95,  # 95% VaR
            'liquidity_buffer': 0.2,  # 20% liquidity buffer
            'min_position_size': 0.001,  # Minimum position size
            'max_leverage': 3.0  # Maximum leverage
        }
        
        # Initialize modules
        self.risk_validator = RiskValidator(self.risk_params)
        self.risk_validator.validate_risk_params()
        
        self.position_manager = PositionManager(self.risk_params)
        self.risk_calculator = RiskCalculator(initial_capital)
        
        # Risk tracking
        self.daily_pnl: List[float] = []
        self.equity_curve: List[float] = [initial_capital]
        self.high_water_mark: float = initial_capital
        self.current_drawdown: float = 0
        self.max_drawdown: float = 0
        self.last_reset: pd.Timestamp = pd.Timestamp.now()
        
        # Correlation matrix
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        
        # Error tracking
        self.calculation_errors = 0
        self.max_calculation_errors = 50
        
        logger.info(f"RiskManager initialized with capital: ${initial_capital:,.2f}")
    
    # Position management delegation
    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        return self.position_manager.positions
    
    @property
    def trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.position_manager.trade_history
    
    @property
    def daily_trades(self) -> int:
        """Get daily trade count"""
        return self.position_manager.daily_trades
    
    @property
    def emergency_mode(self) -> bool:
        """Get emergency mode status"""
        return self.risk_validator.emergency_mode
    
    @property
    def risk_breaches(self) -> List[Dict[str, Any]]:
        """Get risk breaches"""
        return self.risk_validator.risk_breaches
    
    def check_pre_trade_risk(self, symbol: str, side: str, size: float, 
                           price: float, stop_loss: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if trade passes risk criteria with comprehensive error handling.
        
        Args:
            symbol: Trading symbol
            side: Trade side (long/short)
            size: Position size
            price: Entry price
            stop_loss: Optional stop loss price
            
        Returns:
            Tuple of (passed, reason) where passed is boolean and reason is string
        """
        try:
            # Validate inputs
            if not self.position_manager.validate_trade_inputs(symbol, side, size, price):
                return False, "Invalid trade parameters"
            
            # Check calculation errors
            if self.calculation_errors >= self.max_calculation_errors:
                return False, "Too many calculation errors"
            
            # Calculate current metrics
            position_count = self.position_manager.get_position_count()
            total_exposure = self.position_manager.calculate_total_exposure()
            daily_loss = sum(pnl for pnl in self.daily_pnl if pnl < 0)
            
            # Perform risk check
            passed, reason = self.risk_validator.check_pre_trade_risk(
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                stop_loss=stop_loss,
                current_capital=self.current_capital,
                position_count=position_count,
                total_exposure=total_exposure,
                daily_loss=daily_loss
            )
            
            # Check correlation risk
            if passed and not self.risk_validator.check_correlation_risk(
                symbol, self.positions, self.correlation_matrix
            ):
                return False, "Correlation risk too high"
            
            return passed, reason
            
        except Exception as e:
            logger.error(f"Error in pre-trade risk check: {e}")
            logger.error(traceback.format_exc())
            self.calculation_errors += 1
            return False, f"Risk calculation error: {str(e)}"
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """
        Calculate position size based on risk parameters with error handling.
        
        Args:
            entry_price: Entry price for position
            stop_loss: Stop loss price
            symbol: Trading symbol
            
        Returns:
            Calculated position size
        """
        return self.risk_validator.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            current_capital=self.current_capital,
            symbol=symbol,
            current_drawdown=self.current_drawdown,
            existing_positions=self.positions
        )
    
    def add_position(self, symbol: str, side: str, size: float, 
                    entry_price: float, stop_loss: Optional[float] = None) -> None:
        """
        Add new position to portfolio with error handling.
        
        Args:
            symbol: Trading symbol
            side: Position side (long/short)
            size: Position size
            entry_price: Entry price
            stop_loss: Optional stop loss price
        """
        success = self.position_manager.add_position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            current_capital=self.current_capital
        )
        
        if not success:
            self.calculation_errors += 1
    
    def update_position(self, symbol: str, current_price: float) -> None:
        """
        Update position with current price and check stop loss.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        result = self.position_manager.update_position_price(symbol, current_price)
        
        if result.get('success'):
            # Check if stop loss triggered
            if result.get('stop_triggered'):
                logger.info(f"Stop loss triggered for {symbol}")
                self.close_position(symbol, current_price, "Stop loss triggered")
        else:
            self.calculation_errors += 1
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual") -> float:
        """
        Close position and record PnL with error handling.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Realized PnL
        """
        result = self.position_manager.close_position(symbol, exit_price, reason)
        
        if result.get('success'):
            realized_pnl = result['realized_pnl']
            
            # Update capital
            old_capital = self.current_capital
            self.current_capital += realized_pnl
            
            # Capital validation
            if self.current_capital <= 0:
                logger.critical(f"Capital depleted! Current: ${self.current_capital:.2f}")
                self.risk_validator.enter_emergency_mode("capital_depletion")
            
            # Update daily PnL
            self._update_daily_pnl(realized_pnl)
            
            return realized_pnl
        else:
            self.calculation_errors += 1
            return 0
    
    def _update_daily_pnl(self, pnl: float) -> None:
        """Update daily PnL tracking with error handling"""
        try:
            # Validate PnL
            if not np.isfinite(pnl):
                logger.error(f"Invalid PnL value: {pnl}")
                return
            
            self.daily_pnl.append(pnl)
            
            # Update equity curve
            self.equity_curve.append(self.current_capital)
            
            # Update high water mark and drawdown
            if self.current_capital > self.high_water_mark:
                self.high_water_mark = self.current_capital
                self.current_drawdown = 0
            else:
                if self.high_water_mark > 0:
                    self.current_drawdown = (self.high_water_mark - self.current_capital) / self.high_water_mark
                    self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
                    
                    # Check drawdown limit
                    if self.current_drawdown > self.risk_params['max_drawdown']:
                        logger.critical(f"Maximum drawdown exceeded: {self.current_drawdown:.2%}")
                        self.risk_validator.enter_emergency_mode(
                            f"max_drawdown_exceeded: {self.current_drawdown:.2%}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating daily PnL: {e}")
            self.calculation_errors += 1
    
    def calculate_position_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate P&L for a position"""
        if symbol not in self.positions:
            return 0
            
        position = self.positions[symbol]
        
        if position['side'] == 'long':
            return (current_price - position['entry_price']) * position['size']
        else:
            return (position['entry_price'] - current_price) * position['size']
    
    def calculate_risk_limits(self, symbol: str, size: float, direction: int) -> bool:
        """Compatibility method - delegates to check_pre_trade_risk"""
        side = 'long' if direction > 0 else 'short'
        passed, _ = self.check_pre_trade_risk(symbol, side, size, 0)  # Price will be validated separately
        return passed
    
    check_risk_limits = calculate_risk_limits  # Alias for compatibility
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics with error handling.
        
        Returns:
            RiskMetrics object with current metrics
        """
        try:
            # Check calculation errors
            if self.calculation_errors > self.max_calculation_errors:
                logger.error("Too many calculation errors, returning default metrics")
                return self._get_default_risk_metrics()
            
            # Get basic metrics from position manager
            total_exposure = self.position_manager.calculate_total_exposure()
            position_count = self.position_manager.get_position_count()
            largest_position = self.position_manager.get_largest_position()
            unrealized_pnl = self.position_manager.get_unrealized_pnl()
            
            # Calculate PnL
            realized_pnl = self.current_capital - self.initial_capital - unrealized_pnl
            total_pnl = realized_pnl + unrealized_pnl
            
            # Calculate risk metrics using calculator
            var_95_historical = self.risk_calculator.calculate_var(
                self.daily_pnl, 0.95, 1, self.current_capital
            )
            var_95_monte_carlo = self.risk_calculator.calculate_monte_carlo_var(
                self.daily_pnl, 0.95, 1, 1000, self.current_capital
            )
            var_95_parametric = self.risk_calculator.calculate_parametric_var(
                self.daily_pnl, 0.95, 1, self.current_capital
            )
            
            # Use most conservative VaR
            var_95 = max(var_95_historical, var_95_monte_carlo, var_95_parametric)
            cvar_95 = self.risk_calculator.calculate_cvar(
                self.daily_pnl, 0.95, 1, self.current_capital
            )
            sharpe = self.risk_calculator.calculate_sharpe_ratio(self.daily_pnl)
            
            # Calculate risk score
            risk_score = self.risk_calculator.calculate_risk_score(
                total_exposure, position_count, self.current_drawdown, 
                var_95, self.current_capital, self.risk_params
            )
            
            # Add emergency mode penalty
            if self.emergency_mode:
                risk_score = min(100, risk_score + 20)
            
            return RiskMetrics(
                total_exposure=total_exposure,
                position_count=position_count,
                largest_position=largest_position,
                total_pnl=total_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                current_drawdown=self.current_drawdown,
                max_drawdown=self.max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            logger.error(traceback.format_exc())
            return self._get_default_risk_metrics()
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Get default risk metrics when calculation fails"""
        return RiskMetrics(
            total_exposure=0,
            position_count=0,
            largest_position=0,
            total_pnl=0,
            unrealized_pnl=0,
            realized_pnl=0,
            current_drawdown=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            sharpe_ratio=0,
            risk_score=50  # Medium risk by default
        )
    
    def reset_daily_counters(self) -> None:
        """Reset daily counters and limits with error handling"""
        try:
            current_time = pd.Timestamp.now()
            
            # Reset if new day
            if current_time.date() > self.last_reset.date():
                self.daily_pnl = []
                self.position_manager.reset_daily_counters()
                self.last_reset = current_time
                logger.info("Daily risk counters reset")
                
        except Exception as e:
            logger.error(f"Error resetting daily counters: {e}")
    
    def get_risk_adjusted_leverage(self) -> float:
        """
        Calculate appropriate leverage based on current risk.
        
        Returns:
            Recommended leverage multiplier
        """
        return self.risk_validator.get_risk_adjusted_leverage(
            self.current_drawdown, 
            self.position_manager.get_position_count()
        )
    
    def get_available_capital(self) -> float:
        """
        Get available capital for new positions with error handling.
        
        Returns:
            Available capital amount
        """
        try:
            if self.emergency_mode:
                logger.warning("Emergency mode active, no capital available")
                return 0
            
            total_exposure = self.position_manager.calculate_total_exposure()
            reserved_capital = self.initial_capital * self.risk_params['liquidity_buffer']
            
            available = self.current_capital - total_exposure - reserved_capital
            
            # Ensure non-negative
            available = max(0, available)
            
            # Validate result
            if not np.isfinite(available):
                logger.error(f"Invalid available capital calculated: {available}")
                return 0
            
            return available
            
        except Exception as e:
            logger.error(f"Error calculating available capital: {e}")
            return 0
    
    def export_risk_report(self) -> Dict[str, Any]:
        """
        Export comprehensive risk report with error handling.
        
        Returns:
            Dictionary containing full risk analysis
        """
        try:
            metrics = self.calculate_risk_metrics()
            
            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'account': {
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'total_pnl': metrics.total_pnl,
                    'total_pnl_pct': (metrics.total_pnl / self.initial_capital * 100) 
                                     if self.initial_capital > 0 else 0
                },
                'positions': {
                    'count': metrics.position_count,
                    'total_exposure': metrics.total_exposure,
                    'exposure_pct': (metrics.total_exposure / self.current_capital * 100) 
                                   if self.current_capital > 0 else 0,
                    'largest_position': metrics.largest_position,
                    'unrealized_pnl': metrics.unrealized_pnl
                },
                'risk_metrics': {
                    'current_drawdown': metrics.current_drawdown * 100,
                    'max_drawdown': metrics.max_drawdown * 100,
                    'var_95': metrics.var_95,
                    'cvar_95': metrics.cvar_95,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'risk_score': metrics.risk_score
                },
                'limits': {
                    'daily_loss_limit': self.risk_params['max_daily_loss'] * 100,
                    'max_drawdown_limit': self.risk_params['max_drawdown'] * 100,
                    'position_limit': self.risk_params['position_limit'],
                    'daily_trades': self.daily_trades
                },
                'risk_breaches': self.risk_breaches[-10:],  # Last 10 breaches
                'health': {
                    'emergency_mode': self.emergency_mode,
                    'calculation_errors': self.calculation_errors,
                    'leverage': self.get_risk_adjusted_leverage()
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting risk report: {e}")
            return {
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e),
                'emergency_mode': self.emergency_mode
            }
    
    def log_risk_breach(self, breach_type: str, details: str) -> None:
        """
        Log risk limit breach with error handling.
        
        Args:
            breach_type: Type of breach
            details: Breach details
        """
        self.risk_validator.log_risk_breach(breach_type, details)
    
    def save_state(self, filepath: str) -> bool:
        """Save risk manager state to file"""
        try:
            state = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'positions': self.positions,
                'trade_history': self.trade_history[-1000:],  # Last 1000 trades
                'risk_params': self.risk_params,
                'daily_pnl': self.daily_pnl,
                'equity_curve': self.equity_curve[-1000:],  # Last 1000 points
                'high_water_mark': self.high_water_mark,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'risk_breaches': self.risk_breaches[-100:],  # Last 100 breaches
                'emergency_mode': self.emergency_mode
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Risk manager state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving risk manager state: {e}")
            return False
    
    # Delegation methods for VaR calculations (for backward compatibility)
    def calculate_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate Value at Risk - delegates to risk calculator"""
        return self.risk_calculator.calculate_var(
            self.daily_pnl, confidence, horizon, self.current_capital
        )
    
    def calculate_monte_carlo_var(self, confidence: float = 0.95, 
                                 horizon: int = 1, simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR - delegates to risk calculator"""
        return self.risk_calculator.calculate_monte_carlo_var(
            self.daily_pnl, confidence, horizon, simulations, self.current_capital
        )
    
    def calculate_parametric_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate Parametric VaR - delegates to risk calculator"""
        return self.risk_calculator.calculate_parametric_var(
            self.daily_pnl, confidence, horizon, self.current_capital
        )
    
    def calculate_cvar(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate CVaR - delegates to risk calculator"""
        return self.risk_calculator.calculate_cvar(
            self.daily_pnl, confidence, horizon, self.current_capital
        )
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio - delegates to risk calculator"""
        return self.risk_calculator.calculate_sharpe_ratio(self.daily_pnl, risk_free_rate)

"""
REFACTORING SUMMARY:
- Original file: 1700+ lines
- Refactored risk_manager.py: ~550 lines
- Created 3 modular components:
  1. risk_metrics.py - VaR/CVaR calculations and metrics
  2. position_manager.py - Position tracking and management
  3. risk_validator.py - Risk validation and pre-trade checks
- Benefits:
  * Better separation of concerns
  * Easier to test individual components
  * More maintainable code structure
  * Preserved all original functionality
  * Maintained backward compatibility
"""
