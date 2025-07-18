"""
Risk metrics module for trading risk management.
Handles risk calculations, VaR, CVaR, and other metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RiskMetrics:
    """
    Risk metrics data class with validation.
    
    Attributes:
        total_exposure: Total portfolio exposure
        position_count: Number of open positions
        largest_position: Size of largest position
        total_pnl: Total profit/loss
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L
        current_drawdown: Current drawdown percentage
        max_drawdown: Maximum drawdown percentage
        var_95: 95% Value at Risk
        cvar_95: 95% Conditional Value at Risk
        sharpe_ratio: Sharpe ratio
        risk_score: Overall risk score (0-100)
    """
    total_exposure: float
    position_count: int
    largest_position: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    current_drawdown: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    sharpe_ratio: float
    risk_score: float
    
    def __post_init__(self):
        """Validate metrics after initialization"""
        # Ensure all numeric fields are finite
        for field in ['total_exposure', 'largest_position', 'total_pnl', 
                     'unrealized_pnl', 'realized_pnl', 'current_drawdown',
                     'max_drawdown', 'var_95', 'cvar_95', 'sharpe_ratio', 'risk_score']:
            value = getattr(self, field)
            if not np.isfinite(value):
                setattr(self, field, 0.0)
        
        # Ensure percentages are in valid range
        self.current_drawdown = np.clip(self.current_drawdown, 0, 1)
        self.max_drawdown = np.clip(self.max_drawdown, 0, 1)
        self.risk_score = np.clip(self.risk_score, 0, 100)


class RiskCalculator:
    """Handles all risk metric calculations"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        
    def calculate_var(self, daily_pnl: List[float], confidence: float = 0.95, 
                     horizon: int = 1, current_capital: float = None) -> float:
        """
        Calculate Value at Risk with error handling.
        
        Args:
            daily_pnl: List of daily P&L values
            confidence: Confidence level
            horizon: Time horizon in days
            current_capital: Current capital amount
            
        Returns:
            VaR estimate
        """
        try:
            # Validate inputs
            if not 0 < confidence < 1:
                logger.error(f"Invalid confidence level: {confidence}")
                return 0
            
            if horizon <= 0:
                logger.error(f"Invalid horizon: {horizon}")
                return 0
            
            if len(daily_pnl) < 20:
                logger.warning("Insufficient data for VaR calculation")
                return 0
            
            if current_capital is None:
                current_capital = self.initial_capital
            
            # Historical VaR
            returns = pd.Series(daily_pnl) / self.initial_capital
            
            # Remove extreme outliers
            returns = returns[np.abs(returns - returns.mean()) <= 3 * returns.std()]
            
            if len(returns) < 10:
                logger.warning("Too few returns after outlier removal")
                return 0
            
            var_percentile = (1 - confidence) * 100
            var = np.percentile(returns, var_percentile) * current_capital * np.sqrt(horizon)
            
            # Validate result
            if not np.isfinite(var):
                logger.error(f"Invalid VaR calculated: {var}")
                return 0
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0
    
    def calculate_monte_carlo_var(self, daily_pnl: List[float], confidence: float = 0.95, 
                                 horizon: int = 1, simulations: int = 10000,
                                 current_capital: float = None) -> float:
        """
        Calculate Monte Carlo VaR with error handling.
        
        Args:
            daily_pnl: List of daily P&L values
            confidence: Confidence level
            horizon: Time horizon in days
            simulations: Number of simulations
            current_capital: Current capital amount
            
        Returns:
            Monte Carlo VaR estimate
        """
        try:
            # Validate inputs
            if not 0 < confidence < 1 or horizon <= 0 or simulations <= 0:
                logger.error("Invalid Monte Carlo VaR parameters")
                return 0
            
            if len(daily_pnl) < 20:
                return 0
            
            if current_capital is None:
                current_capital = self.initial_capital
            
            # Cap simulations to prevent memory issues
            simulations = min(simulations, 100000)
            
            returns = np.array(daily_pnl) / self.initial_capital
            
            # Remove extreme outliers
            returns = returns[np.abs(returns - np.mean(returns)) <= 3 * np.std(returns)]
            
            if len(returns) < 10:
                return 0
            
            # Generate random scenarios
            try:
                simulated_returns = np.random.choice(
                    returns, 
                    size=(simulations, horizon), 
                    replace=True
                )
                portfolio_returns = np.sum(simulated_returns, axis=1)
            except MemoryError:
                logger.error("Memory error in Monte Carlo simulation, reducing simulations")
                simulations = 1000
                simulated_returns = np.random.choice(
                    returns, 
                    size=(simulations, horizon), 
                    replace=True
                )
                portfolio_returns = np.sum(simulated_returns, axis=1)
            
            # Calculate VaR
            var_percentile = (1 - confidence) * 100
            monte_carlo_var = np.percentile(portfolio_returns, var_percentile) * current_capital
            
            # Validate result
            if not np.isfinite(monte_carlo_var):
                logger.error(f"Invalid Monte Carlo VaR: {monte_carlo_var}")
                return 0
            
            return abs(monte_carlo_var)
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR: {e}")
            return 0
    
    def calculate_parametric_var(self, daily_pnl: List[float], confidence: float = 0.95, 
                                horizon: int = 1, current_capital: float = None) -> float:
        """
        Calculate Parametric VaR with error handling.
        
        Args:
            daily_pnl: List of daily P&L values
            confidence: Confidence level
            horizon: Time horizon in days
            current_capital: Current capital amount
            
        Returns:
            Parametric VaR estimate
        """
        try:
            if not 0 < confidence < 1 or horizon <= 0:
                return 0
            
            if len(daily_pnl) < 20:
                return 0
            
            if current_capital is None:
                current_capital = self.initial_capital
            
            returns = pd.Series(daily_pnl) / self.initial_capital
            
            # Remove extreme outliers
            returns = returns[np.abs(returns - returns.mean()) <= 3 * returns.std()]
            
            if len(returns) < 10:
                return 0
            
            # Calculate parameters
            mean_return = returns.mean()
            volatility = returns.std()
            
            if volatility <= 0:
                logger.warning("Zero volatility calculated")
                return 0
            
            # Normal distribution critical value
            try:
                z_score = norm.ppf(1 - confidence)
            except Exception:
                z_score = -1.645  # Default for 95% confidence
            
            # Parametric VaR
            var = (mean_return + z_score * volatility) * current_capital * np.sqrt(horizon)
            
            if not np.isfinite(var):
                return 0
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error in parametric VaR: {e}")
            return 0
    
    def calculate_cvar(self, daily_pnl: List[float], confidence: float = 0.95, 
                      horizon: int = 1, current_capital: float = None) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall) with error handling.
        
        Args:
            daily_pnl: List of daily P&L values
            confidence: Confidence level
            horizon: Time horizon in days
            current_capital: Current capital amount
            
        Returns:
            CVaR estimate
        """
        try:
            if not 0 < confidence < 1 or horizon <= 0:
                return 0
            
            if len(daily_pnl) < 20:
                return 0
            
            if current_capital is None:
                current_capital = self.initial_capital
            
            returns = pd.Series(daily_pnl) / self.initial_capital
            
            # Remove extreme outliers
            returns = returns[np.abs(returns - returns.mean()) <= 3 * returns.std()]
            
            if len(returns) < 10:
                return 0
            
            # Get returns below VaR threshold
            var_percentile = (1 - confidence) * 100
            var_threshold = np.percentile(returns, var_percentile)
            
            # Calculate average of returns below threshold
            tail_returns = returns[returns <= var_threshold]
            
            if len(tail_returns) > 0:
                cvar = tail_returns.mean() * current_capital * np.sqrt(horizon)
                
                if np.isfinite(cvar):
                    return abs(cvar)
            
            # Fallback to VaR
            return self.calculate_var(daily_pnl, confidence, horizon, current_capital)
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0
    
    def calculate_sharpe_ratio(self, daily_pnl: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio with error handling.
        
        Args:
            daily_pnl: List of daily P&L values
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        try:
            if len(daily_pnl) < 20:
                return 0
            
            # Validate risk-free rate
            if not 0 <= risk_free_rate <= 0.2:
                logger.warning(f"Unusual risk-free rate: {risk_free_rate}")
                risk_free_rate = 0.02
            
            returns = pd.Series(daily_pnl) / self.initial_capital
            
            # Remove extreme outliers
            returns = returns[np.abs(returns - returns.mean()) <= 3 * returns.std()]
            
            if len(returns) < 10:
                return 0
            
            # Annualized metrics
            daily_return = returns.mean()
            annual_return = daily_return * 252
            
            daily_vol = returns.std()
            
            if daily_vol <= 0:
                return 0
            
            annual_vol = daily_vol * np.sqrt(252)
            
            sharpe = (annual_return - risk_free_rate) / annual_vol
            
            # Cap Sharpe ratio to reasonable range
            sharpe = np.clip(sharpe, -3, 3)
            
            if not np.isfinite(sharpe):
                return 0
            
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def calculate_risk_score(self, exposure: float, positions: int, drawdown: float, 
                           var: float, current_capital: float, risk_params: Dict) -> float:
        """
        Calculate overall risk score with error handling.
        
        Args:
            exposure: Total exposure
            positions: Number of positions
            drawdown: Current drawdown
            var: Value at Risk
            current_capital: Current capital
            risk_params: Risk parameters
            
        Returns:
            Risk score (0-100)
        """
        try:
            # Exposure risk (0-30 points)
            if current_capital > 0:
                exposure_pct = exposure / current_capital
                exposure_score = min(30, exposure_pct * 30 / risk_params['max_total_exposure'])
            else:
                exposure_score = 30
            
            # Position concentration (0-20 points)
            concentration_score = min(20, positions * 20 / risk_params['position_limit'])
            
            # Drawdown risk (0-30 points)
            drawdown_score = min(30, drawdown * 30 / risk_params['max_drawdown'])
            
            # VaR risk (0-20 points)
            if current_capital > 0:
                var_pct = var / current_capital
                var_score = min(20, var_pct * 20 / 0.05)
            else:
                var_score = 20
            
            total_score = exposure_score + concentration_score + drawdown_score + var_score
            
            return min(100, total_score)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50  # Default medium risk
