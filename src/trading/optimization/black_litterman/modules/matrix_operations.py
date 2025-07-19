"""
Matrix operations module for Black-Litterman optimization.
Handles covariance matrix calculations, shrinkage, and numerical stability.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class MatrixOperations:
    """Handles all matrix operations for Black-Litterman optimization"""
    
    def __init__(self, min_variance: float = 1e-8, numerical_tolerance: float = 1e-10):
        self.min_variance = min_variance
        self.numerical_tolerance = numerical_tolerance
        
    def calculate_covariance_matrix(self,
                                  returns_data: pd.DataFrame,
                                  method: str = 'shrinkage') -> pd.DataFrame:
        """
        Calculate covariance matrix with various methods.
        
        Args:
            returns_data: DataFrame of asset returns
            method: Covariance estimation method ('shrinkage', 'exponential', 'simple')
            
        Returns:
            Covariance matrix as DataFrame
        """
        try:
            # Remove assets with insufficient data
            valid_assets = []
            for col in returns_data.columns:
                if returns_data[col].notna().sum() > 10:  # At least 10 observations
                    valid_assets.append(col)
                else:
                    logger.warning(f"Removing {col} due to insufficient data")
            
            if not valid_assets:
                raise ValueError("No assets with sufficient data")
                
            returns_data = returns_data[valid_assets]
            
            if method == 'exponential':
                # Exponentially weighted covariance
                cov = returns_data.ewm(span=60, min_periods=20).cov()
                # Get the most recent covariance
                if isinstance(cov.index, pd.MultiIndex):
                    last_date = returns_data.index[-1]
                    cov = cov.loc[last_date]
            elif method == 'shrinkage':
                # Ledoit-Wolf shrinkage
                cov = self.ledoit_wolf_shrinkage(returns_data)
            else:
                # Simple sample covariance
                cov = returns_data.cov()
            
            # Ensure minimum variance
            cov = cov.clip(lower=self.min_variance)
            
            return cov
            
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            # Return diagonal matrix as fallback
            n = len(returns_data.columns)
            variances = returns_data.var().fillna(0.01)  # Default 1% variance
            return pd.DataFrame(
                np.diag(variances),
                index=returns_data.columns,
                columns=returns_data.columns
            )
    
    def ledoit_wolf_shrinkage(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator for covariance matrix.
        
        This method shrinks the sample covariance matrix towards a structured
        estimator (scaled identity matrix) to improve out-of-sample performance.
        
        Args:
            returns_data: DataFrame of asset returns
            
        Returns:
            Shrunk covariance matrix as DataFrame
        """
        try:
            T, N = returns_data.shape
            
            # Check data sufficiency
            if T < 2:
                logger.warning("Not enough observations for covariance estimation")
                return returns_data.cov()
            
            # Handle missing values
            returns_clean = returns_data.fillna(0)
            
            # Sample covariance matrix
            S = returns_clean.cov().values
            
            # Ensure positive variances
            np.fill_diagonal(S, np.maximum(np.diagonal(S), self.min_variance))
            
            # Shrinkage target (identity matrix scaled by average variance)
            mu = np.trace(S) / N
            F = mu * np.eye(N)
            
            # Calculate shrinkage intensity
            X = returns_clean.values
            X_centered = X - X.mean(axis=0)
            
            # Safe shrinkage calculation
            try:
                # Shrinkage intensity calculation (simplified)
                sample_cov = X_centered.T @ X_centered / T
                
                # Frobenius norm of covariance
                norm_sample = np.linalg.norm(sample_cov - F, 'fro')
                
                # Calculate optimal shrinkage
                if norm_sample > 0:
                    # Simplified shrinkage formula
                    mse = np.mean((sample_cov - S) ** 2)
                    shrinkage = min(1, max(0, mse / (norm_sample ** 2)))
                else:
                    shrinkage = 0
                    
            except Exception as e:
                logger.warning(f"Error in shrinkage calculation: {e}, using default")
                shrinkage = 0.1  # Default shrinkage
            
            # Shrunk covariance matrix
            shrunk_cov = shrinkage * F + (1 - shrinkage) * S
            
            # Ensure positive definiteness
            shrunk_cov = self.ensure_positive_definite(shrunk_cov)
            
            return pd.DataFrame(
                shrunk_cov,
                index=returns_data.columns,
                columns=returns_data.columns
            )
            
        except Exception as e:
            logger.error(f"Error in Ledoit-Wolf shrinkage: {e}")
            return returns_data.cov()
    
    def is_positive_definite(self, matrix: pd.DataFrame) -> bool:
        """Check if matrix is positive definite"""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite"""
        try:
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            
            # Set minimum eigenvalue
            min_eigenvalue = self.min_variance
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            
            # Reconstruct matrix
            matrix_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            return matrix_fixed
            
        except Exception as e:
            logger.error(f"Error ensuring positive definite: {e}")
            # Return diagonal matrix as fallback
            return np.diag(np.diagonal(matrix))
    
    def fix_covariance_matrix(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Fix a non-positive definite covariance matrix"""
        try:
            values_fixed = self.ensure_positive_definite(cov_matrix.values)
            return pd.DataFrame(
                values_fixed,
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )
        except Exception as e:
            logger.error(f"Error fixing covariance matrix: {e}")
            # Return diagonal matrix
            variances = np.diagonal(cov_matrix)
            return pd.DataFrame(
                np.diag(variances),
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )
    
    def safe_inverse(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        """Safely compute matrix inverse"""
        try:
            # Try standard inverse
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            try:
                # Try pseudo-inverse
                return np.linalg.pinv(matrix)
            except Exception as e:
                logger.error(f"Failed to compute matrix inverse: {e}")
                return None
    
    def calculate_portfolio_metrics(self,
                                  weights: np.ndarray,
                                  returns: np.ndarray,
                                  covariance: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio metrics given weights, returns, and covariance.
        
        Args:
            weights: Portfolio weights
            returns: Expected returns
            covariance: Covariance matrix
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        try:
            # Portfolio expected return
            expected_return = np.dot(weights, returns)
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            portfolio_volatility = np.sqrt(max(0, portfolio_variance))
            
            # Sharpe ratio (assuming zero risk-free rate)
            if portfolio_volatility > self.numerical_tolerance:
                sharpe_ratio = expected_return / portfolio_volatility
            else:
                sharpe_ratio = 0
                logger.warning("Portfolio volatility too low for Sharpe calculation")
            
            return expected_return, portfolio_volatility, sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return 0, 0, 0
