"""
Black-Litterman Portfolio Optimizer - Main coordination module
Advanced portfolio optimization for crypto assets using the Black-Litterman model.

File: black_litterman.py
Modified: 2024-12-19
Refactored: 2025-01-18
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import traceback
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import modularized components
from .modules.matrix_operations import MatrixOperations
from .modules.bayesian_updater import BayesianUpdater
from .modules.portfolio_optimizer import PortfolioOptimizer
from .modules.view_generator import CryptoViewGenerator

from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class BlackLittermanOptimizer:
    """
    Advanced Black-Litterman portfolio optimization for crypto assets.
    
    The Black-Litterman model combines market equilibrium with investor views
    to produce optimal portfolio weights. This implementation includes:
    - Covariance matrix shrinkage (Ledoit-Wolf)
    - Multiple constraint types
    - Fallback mechanisms for numerical stability
    
    This is the main coordination class that delegates to specialized modules:
    - matrix_operations: Covariance calculations and matrix operations
    - bayesian_updater: Incorporating investor views
    - portfolio_optimizer: Weight optimization with constraints
    - view_generator: Generating views from various sources
    
    Attributes:
        risk_aversion (float): Risk aversion parameter (default: 3.0)
        tau (float): Uncertainty in prior estimate (default: 0.025)
    """
    
    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.025):
        """
        Initialize the Black-Litterman optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter (higher = more conservative)
            tau: Uncertainty in prior estimate (typically 0.025-0.05)
        """
        # Validate parameters
        if not isinstance(risk_aversion, (int, float)) or risk_aversion <= 0:
            logger.warning(f"Invalid risk_aversion {risk_aversion}, using default 3.0")
            risk_aversion = 3.0
            
        if not isinstance(tau, (int, float)) or tau <= 0 or tau > 1:
            logger.warning(f"Invalid tau {tau}, using default 0.025")
            tau = 0.025
            
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # Numerical stability parameters
        self.min_variance = 1e-8
        self.max_weight = 0.95
        self.min_weight = 0.0
        self.numerical_tolerance = 1e-10
        
        # Initialize module managers
        self.matrix_ops = MatrixOperations(
            min_variance=self.min_variance,
            numerical_tolerance=self.numerical_tolerance
        )
        self.bayesian_updater = BayesianUpdater(
            tau=tau,
            min_variance=self.min_variance
        )
        self.portfolio_optimizer = PortfolioOptimizer(
            risk_aversion=risk_aversion,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            numerical_tolerance=self.numerical_tolerance
        )
    
    def optimize_portfolio(self,
                         returns_data: pd.DataFrame,
                         market_caps: pd.Series,
                         views: Optional[Dict] = None,
                         view_confidences: Optional[Dict] = None,
                         constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio using Black-Litterman model.
        
        Args:
            returns_data: DataFrame of asset returns (rows: time, columns: assets)
            market_caps: Series of market capitalizations (index: assets)
            views: Dict of investor views {asset: expected_return}
            view_confidences: Dict of confidence levels {asset: confidence}
            constraints: Additional portfolio constraints
            
        Returns:
            Dict containing:
                - weights: Optimal portfolio weights
                - expected_returns: Black-Litterman expected returns
                - covariance: Black-Litterman covariance matrix
                - expected_portfolio_return: Expected portfolio return
                - portfolio_volatility: Portfolio standard deviation
                - sharpe_ratio: Portfolio Sharpe ratio
                - market_weights: Market equilibrium weights
                - equilibrium_returns: Implied equilibrium returns
        """
        try:
            # Validate inputs
            if returns_data is None or returns_data.empty:
                raise ValueError("Returns data is empty")
                
            if market_caps is None or market_caps.empty:
                raise ValueError("Market caps data is empty")
            
            # Ensure alignment
            common_assets = returns_data.columns.intersection(market_caps.index)
            if len(common_assets) == 0:
                raise ValueError("No common assets between returns and market caps")
                
            if len(common_assets) < len(returns_data.columns):
                logger.warning(f"Only {len(common_assets)} assets have market cap data")
                returns_data = returns_data[common_assets]
                market_caps = market_caps[common_assets]
            
            # Check for sufficient data
            min_observations = max(20, len(returns_data.columns) * 2)
            if len(returns_data) < min_observations:
                logger.warning(f"Insufficient data: {len(returns_data)} rows < {min_observations} recommended")
            
            # Step 1: Calculate market portfolio weights (equilibrium)
            market_weights = self.portfolio_optimizer.calculate_market_weights(market_caps)
            
            # Step 2: Calculate covariance matrix with shrinkage
            cov_matrix = self.matrix_ops.calculate_covariance_matrix(
                returns_data, method='shrinkage'
            )
            
            # Validate covariance matrix
            if not self.matrix_ops.is_positive_definite(cov_matrix):
                logger.warning("Covariance matrix not positive definite, applying fix")
                cov_matrix = self.matrix_ops.fix_covariance_matrix(cov_matrix)
            
            # Step 3: Calculate implied equilibrium returns
            equilibrium_returns = self.portfolio_optimizer.calculate_equilibrium_returns(
                market_weights, cov_matrix
            )
            
            # Step 4: Incorporate investor views
            if views is not None and view_confidences is not None:
                # Validate views
                views = self.bayesian_updater.validate_views(views, returns_data.columns)
                view_confidences = self.bayesian_updater.validate_view_confidences(
                    view_confidences, views
                )
                
                if views:
                    bl_returns, bl_cov = self.bayesian_updater.incorporate_views(
                        equilibrium_returns, cov_matrix, views, view_confidences,
                        self.matrix_ops
                    )
                else:
                    logger.warning("No valid views provided, using equilibrium returns")
                    bl_returns = equilibrium_returns
                    bl_cov = cov_matrix
            else:
                bl_returns = equilibrium_returns
                bl_cov = cov_matrix
            
            # Step 5: Optimize portfolio with constraints
            optimal_weights = self.portfolio_optimizer.optimize_weights(
                bl_returns, bl_cov, constraints
            )
            
            # Validate final weights
            optimal_weights = self.portfolio_optimizer.validate_weights(optimal_weights)
            
            # Calculate portfolio metrics
            expected_return, portfolio_volatility, sharpe_ratio = \
                self.matrix_ops.calculate_portfolio_metrics(
                    optimal_weights.values, bl_returns.values, bl_cov.values
                )
            
            return {
                'weights': optimal_weights,
                'expected_returns': bl_returns,
                'covariance': bl_cov,
                'expected_portfolio_return': expected_return,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'market_weights': market_weights,
                'equilibrium_returns': equilibrium_returns,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback to equal weights
            n_assets = len(returns_data.columns)
            fallback_weights = pd.Series(
                np.ones(n_assets) / n_assets,
                index=returns_data.columns
            )
            
            # Calculate basic statistics
            simple_returns = returns_data.mean()
            simple_cov = returns_data.cov()
            simple_volatility = np.sqrt(np.diagonal(simple_cov).mean())
            
            return {
                'weights': fallback_weights,
                'expected_returns': simple_returns,
                'covariance': simple_cov,
                'expected_portfolio_return': simple_returns.mean(),
                'portfolio_volatility': simple_volatility,
                'sharpe_ratio': simple_returns.mean() / simple_volatility if simple_volatility > 0 else 0,
                'status': 'fallback',
                'error': str(e)
            }
    
    def create_view_generator(self,
                            ml_models: Optional[Dict] = None,
                            market_data: Optional[Dict] = None) -> CryptoViewGenerator:
        """
        Create a view generator instance for generating investor views.
        
        Args:
            ml_models: Dict of ML models {symbol: model}
            market_data: Dict of market data {symbol: DataFrame}
            
        Returns:
            CryptoViewGenerator instance
        """
        return CryptoViewGenerator(ml_models, market_data)
    
    def update_risk_aversion(self, new_risk_aversion: float):
        """
        Update the risk aversion parameter.
        
        Args:
            new_risk_aversion: New risk aversion value
        """
        if isinstance(new_risk_aversion, (int, float)) and new_risk_aversion > 0:
            self.risk_aversion = new_risk_aversion
            self.portfolio_optimizer.risk_aversion = new_risk_aversion
            logger.info(f"Updated risk aversion to {new_risk_aversion}")
        else:
            logger.warning(f"Invalid risk aversion {new_risk_aversion}, keeping current value")
    
    def update_tau(self, new_tau: float):
        """
        Update the tau parameter (uncertainty in prior).
        
        Args:
            new_tau: New tau value
        """
        if isinstance(new_tau, (int, float)) and 0 < new_tau <= 1:
            self.tau = new_tau
            self.bayesian_updater.tau = new_tau
            logger.info(f"Updated tau to {new_tau}")
        else:
            logger.warning(f"Invalid tau {new_tau}, keeping current value")


# Export the view generator class as well
__all__ = ['BlackLittermanOptimizer', 'CryptoViewGenerator']

"""
MODULARIZATION SUMMARY:
- Original file: 1,500+ lines
- Refactored main file: ~300 lines
- Modules created:
  1. matrix_operations.py (~300 lines) - Matrix calculations and numerical stability
  2. bayesian_updater.py (~250 lines) - Bayesian view incorporation
  3. portfolio_optimizer.py (~400 lines) - Weight optimization and constraints
  4. view_generator.py (~400 lines) - View generation from various sources
  
Benefits:
- Clear separation of mathematical operations
- Isolated Bayesian updating logic
- Modular optimization algorithms
- Reusable view generation
- Better testability of individual components
"""
