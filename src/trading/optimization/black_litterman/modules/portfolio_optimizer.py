"""
Portfolio optimizer module for Black-Litterman optimization.
Handles weight optimization with various constraints.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import logging
from typing import Dict, Optional
import traceback

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Optimizes portfolio weights given expected returns and covariance"""
    
    def __init__(self, risk_aversion: float = 3.0,
                 min_weight: float = 0.0,
                 max_weight: float = 0.95,
                 numerical_tolerance: float = 1e-10):
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.numerical_tolerance = numerical_tolerance
        
    def optimize_weights(self,
                        expected_returns: pd.Series,
                        cov_matrix: pd.DataFrame,
                        constraints: Optional[Dict] = None) -> pd.Series:
        """
        Optimize portfolio weights with constraints.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            constraints: Portfolio constraints dict
            
        Returns:
            Series of optimal portfolio weights
        """
        try:
            n_assets = len(expected_returns)
            
            # Validate inputs
            if n_assets == 0:
                raise ValueError("No assets to optimize")
            
            # Default constraints for crypto
            default_constraints = {
                'max_weight': 0.4,  # Max 40% per asset
                'min_weight': 0.0,  # No short selling
                'max_concentration': 0.6,  # Max 60% in top 3 assets
                'min_diversification': 3  # Minimum 3 assets
            }
            
            if constraints:
                default_constraints.update(constraints)
            
            # Validate constraint values
            default_constraints['max_weight'] = min(
                self.max_weight, 
                max(0.01, default_constraints['max_weight'])
            )
            default_constraints['min_weight'] = max(
                self.min_weight,
                min(0, default_constraints['min_weight'])
            )
            
            # Objective function: maximize utility
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                # Add small penalty for numerical stability
                penalty = self.numerical_tolerance * np.sum(weights ** 2)
                return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance) + penalty
            
            # Constraints
            constraint_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Add constraints carefully
            if n_assets >= 3 and 'max_concentration' in default_constraints:
                max_conc = default_constraints['max_concentration']
                
                def concentration_constraint(weights):
                    sorted_weights = np.sort(weights)[::-1]  # Descending order
                    top_3_weight = np.sum(sorted_weights[:min(3, n_assets)])
                    return max_conc - top_3_weight
                
                constraint_list.append({
                    'type': 'ineq',
                    'fun': concentration_constraint
                })
            
            # Minimum diversification
            if n_assets >= default_constraints.get('min_diversification', 3):
                min_assets = default_constraints['min_diversification']
                
                def diversification_constraint(weights):
                    active_assets = np.sum(weights > 0.01)  # Assets with >1% weight
                    return active_assets - min_assets
                
                constraint_list.append({
                    'type': 'ineq',
                    'fun': diversification_constraint
                })
            
            # Bounds
            min_weight = default_constraints.get('min_weight', 0.0)
            max_weight = default_constraints.get('max_weight', 0.4)
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # Initial guess (equal weights with small random perturbation)
            x0 = np.ones(n_assets) / n_assets
            x0 += np.random.normal(0, 0.01, n_assets)
            x0 = np.maximum(min_weight, np.minimum(max_weight, x0))
            x0 = x0 / np.sum(x0)  # Normalize
            
            # Try multiple optimization methods
            methods = ['SLSQP', 'trust-constr']
            result = None
            
            for method in methods:
                try:
                    if method == 'trust-constr':
                        # Convert constraints for trust-constr
                        # Sum to 1 constraint
                        A_eq = np.ones((1, n_assets))
                        linear_constraint = LinearConstraint(A_eq, 1, 1)
                        
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            constraints=[linear_constraint]
                        )
                    else:
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            constraints=constraint_list,
                            options={'maxiter': 1000}
                        )
                    
                    if result.success:
                        break
                        
                except Exception as e:
                    logger.warning(f"Optimization failed with {method}: {e}")
                    continue
            
            if result and result.success:
                weights = result.x
                # Ensure weights are valid
                weights = self.validate_weights_array(weights)
                return pd.Series(weights, index=expected_returns.index)
            else:
                logger.warning("Optimization failed, using fallback weights")
                return self.fallback_weights(expected_returns, default_constraints)
                
        except Exception as e:
            logger.error(f"Error in weight optimization: {e}")
            logger.error(traceback.format_exc())
            return self.fallback_weights(expected_returns, constraints or {})
    
    def calculate_market_weights(self, market_caps: pd.Series) -> pd.Series:
        """
        Calculate market capitalization weights.
        
        Args:
            market_caps: Series of market capitalizations
            
        Returns:
            Series of normalized market weights
        """
        try:
            # Remove invalid market caps
            valid_caps = market_caps[market_caps > 0]
            
            if len(valid_caps) == 0:
                logger.warning("No valid market caps, using equal weights")
                return pd.Series(
                    np.ones(len(market_caps)) / len(market_caps),
                    index=market_caps.index
                )
            
            total_market_cap = valid_caps.sum()
            
            if total_market_cap > 0:
                weights = pd.Series(0.0, index=market_caps.index)
                weights[valid_caps.index] = valid_caps / total_market_cap
                return weights
            else:
                # Equal weights fallback
                return pd.Series(
                    np.ones(len(market_caps)) / len(market_caps),
                    index=market_caps.index
                )
                
        except Exception as e:
            logger.error(f"Error calculating market weights: {e}")
            # Equal weights fallback
            return pd.Series(
                np.ones(len(market_caps)) / len(market_caps),
                index=market_caps.index
            )
    
    def calculate_equilibrium_returns(self,
                                    market_weights: pd.Series,
                                    cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate implied equilibrium returns using reverse optimization.
        
        Args:
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix
            
        Returns:
            Series of equilibrium expected returns
        """
        try:
            # Validate inputs
            if market_weights.sum() == 0:
                raise ValueError("Market weights sum to zero")
            
            # Ensure alignment
            common_assets = market_weights.index.intersection(cov_matrix.index)
            market_weights = market_weights[common_assets]
            cov_matrix = cov_matrix.loc[common_assets, common_assets]
            
            # Reverse optimization: mu = lambda * Sigma * w
            equilibrium_returns = self.risk_aversion * np.dot(cov_matrix, market_weights)
            
            # Validate returns
            if not np.all(np.isfinite(equilibrium_returns)):
                logger.warning("Non-finite equilibrium returns detected")
                equilibrium_returns = np.nan_to_num(equilibrium_returns, 0)
            
            return pd.Series(equilibrium_returns, index=common_assets)
            
        except Exception as e:
            logger.error(f"Error calculating equilibrium returns: {e}")
            # Return zero returns as fallback
            return pd.Series(0, index=market_weights.index)
    
    def fallback_weights(self,
                        expected_returns: pd.Series,
                        constraints: Dict) -> pd.Series:
        """
        Fallback weight calculation if optimization fails.
        
        Args:
            expected_returns: Expected returns for each asset
            constraints: Portfolio constraints dict
            
        Returns:
            Series of fallback portfolio weights
        """
        try:
            n_assets = len(expected_returns)
            max_weight = constraints.get('max_weight', 0.4)
            
            # Try risk parity approach
            if hasattr(expected_returns, 'index'):
                # Use inverse volatility if we have the data
                weights = np.ones(n_assets) / n_assets
            else:
                weights = np.ones(n_assets) / n_assets
            
            # Cap maximum weights
            weights = np.minimum(weights, max_weight)
            weights = weights / np.sum(weights)
            
            return pd.Series(weights, index=expected_returns.index)
            
        except Exception as e:
            logger.error(f"Error in fallback weights: {e}")
            # Ultimate fallback: equal weights
            n = len(expected_returns)
            return pd.Series(np.ones(n) / n, index=expected_returns.index)
    
    def validate_weights(self, weights: pd.Series) -> pd.Series:
        """Validate and normalize portfolio weights"""
        try:
            # Remove any NaN or infinite values
            weights = weights.fillna(0)
            weights[~np.isfinite(weights)] = 0
            
            # Ensure non-negative
            weights = weights.clip(lower=0)
            
            # Normalize to sum to 1
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                # Equal weights if all zero
                weights = pd.Series(
                    np.ones(len(weights)) / len(weights),
                    index=weights.index
                )
            
            return weights
            
        except Exception as e:
            logger.error(f"Error validating weights: {e}")
            # Return equal weights
            n = len(weights)
            return pd.Series(np.ones(n) / n, index=weights.index)
    
    def validate_weights_array(self, weights: np.ndarray) -> np.ndarray:
        """Validate and normalize weight array"""
        # Ensure non-negative
        weights = np.maximum(weights, 0)
        
        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights
