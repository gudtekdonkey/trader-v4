"""
Data preprocessing module for Hierarchical Risk Parity optimization.
Handles returns data cleaning, validation, and correlation calculations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for HRP optimization"""
    
    def __init__(self, min_variance: float = 1e-8, numerical_tolerance: float = 1e-10):
        self.min_variance = min_variance
        self.numerical_tolerance = numerical_tolerance
        
    def clean_returns_data(self, returns: pd.DataFrame, missing_threshold: float = 0.5) -> pd.DataFrame:
        """
        Clean returns data by handling missing values and outliers.
        
        Args:
            returns: DataFrame of asset returns
            missing_threshold: Maximum allowed missing data percentage per asset
            
        Returns:
            Cleaned returns DataFrame
        """
        try:
            # Remove assets with too many missing values
            valid_assets = []
            
            for col in returns.columns:
                missing_pct = returns[col].isna().sum() / len(returns)
                if missing_pct <= missing_threshold:
                    valid_assets.append(col)
                else:
                    logger.warning(f"Removing {col} due to {missing_pct:.1%} missing data")
            
            if not valid_assets:
                raise ValueError("No assets with sufficient data")
            
            returns_clean = returns[valid_assets].copy()
            
            # Handle remaining missing values
            # Forward fill then backward fill
            returns_clean = returns_clean.fillna(method='ffill').fillna(method='bfill')
            
            # If still NaN, fill with zeros
            returns_clean = returns_clean.fillna(0)
            
            # Cap extreme outliers
            for col in returns_clean.columns:
                # Cap at 5 standard deviations
                std = returns_clean[col].std()
                mean = returns_clean[col].mean()
                if std > 0:
                    returns_clean[col] = returns_clean[col].clip(
                        lower=mean - 5*std,
                        upper=mean + 5*std
                    )
            
            return returns_clean
            
        except Exception as e:
            logger.error(f"Error cleaning returns data: {e}")
            # Return original data if cleaning fails
            return returns.fillna(0)
    
    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix with error handling.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Correlation matrix as DataFrame
        """
        try:
            # Calculate correlation
            correlation = returns.corr()
            
            # Check for NaN values
            if correlation.isna().any().any():
                logger.warning("NaN values in correlation matrix, using pairwise correlation")
                correlation = returns.corr(method='pearson', min_periods=10)
                
                # Fill remaining NaN with 0
                correlation = correlation.fillna(0)
                
                # Set diagonal to 1
                np.fill_diagonal(correlation.values, 1.0)
            
            # Ensure symmetry
            correlation = (correlation + correlation.T) / 2
            
            # Ensure valid correlation values
            correlation = correlation.clip(lower=-1, upper=1)
            
            # Ensure diagonal is exactly 1
            np.fill_diagonal(correlation.values, 1.0)
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            # Return identity matrix as fallback
            n = len(returns.columns)
            return pd.DataFrame(
                np.eye(n),
                index=returns.columns,
                columns=returns.columns
            )
    
    def calculate_distance_matrix(self, correlation_matrix: pd.DataFrame, 
                                distance_metric: str = 'correlation') -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.
        
        Args:
            correlation_matrix: Correlation matrix
            distance_metric: Type of distance metric ('correlation' or 'angular')
            
        Returns:
            Distance matrix as numpy array
        """
        try:
            # Validate correlation matrix
            if correlation_matrix.empty:
                raise ValueError("Empty correlation matrix")
            
            # Ensure correlation values are in valid range
            corr_values = correlation_matrix.values.clip(-1, 1)
            
            if distance_metric == 'correlation':
                # Distance = sqrt(0.5 * (1 - correlation))
                distance_matrix = np.sqrt(np.maximum(0, 0.5 * (1 - corr_values)))
            elif distance_metric == 'angular':
                # Angular distance
                distance_matrix = np.sqrt(np.maximum(0, 2 * (1 - corr_values)))
            else:
                # Default to correlation distance
                distance_matrix = np.sqrt(np.maximum(0, 0.5 * (1 - corr_values)))
            
            # Ensure diagonal is zero
            np.fill_diagonal(distance_matrix, 0)
            
            # Ensure symmetry
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(distance_matrix)):
                logger.warning("Non-finite values in distance matrix")
                distance_matrix = np.nan_to_num(distance_matrix, nan=0, posinf=1, neginf=0)
            
            return distance_matrix
            
        except Exception as e:
            logger.error(f"Error calculating distance matrix: {e}")
            # Return identity-based distance matrix as fallback
            n = len(correlation_matrix)
            distance_matrix = np.ones((n, n)) * 0.5
            np.fill_diagonal(distance_matrix, 0)
            return distance_matrix
    
    def validate_returns_data(self, returns: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate returns data for HRP optimization.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if returns is None or returns.empty:
                return False, "Returns data is empty"
                
            if len(returns.columns) < 2:
                return False, "Need at least 2 assets for HRP optimization"
                
            if len(returns) < 10:
                logger.warning(f"Limited data: only {len(returns)} observations")
                
            # Check for all NaN columns
            for col in returns.columns:
                if returns[col].isna().all():
                    return False, f"Asset {col} has all NaN values"
            
            # Check for zero variance assets
            for col in returns.columns:
                if returns[col].std() == 0:
                    logger.warning(f"Asset {col} has zero variance")
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame, 
                                  correlation_matrix: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate covariance matrix from returns or correlation matrix.
        
        Args:
            returns: DataFrame of asset returns
            correlation_matrix: Pre-calculated correlation matrix (optional)
            
        Returns:
            Covariance matrix as DataFrame
        """
        try:
            if correlation_matrix is None:
                # Direct covariance calculation
                covariance = returns.cov()
            else:
                # Convert correlation to covariance using standard deviations
                stds = returns.std()
                
                # Handle zero standard deviations
                stds = stds.replace(0, self.min_variance)
                
                # Covariance = Correlation * std_i * std_j
                covariance = correlation_matrix.copy()
                for i in range(len(stds)):
                    for j in range(len(stds)):
                        covariance.iloc[i, j] = correlation_matrix.iloc[i, j] * stds.iloc[i] * stds.iloc[j]
            
            # Ensure minimum variance on diagonal
            diagonal = np.diagonal(covariance.values)
            diagonal = np.maximum(diagonal, self.min_variance)
            np.fill_diagonal(covariance.values, diagonal)
            
            return covariance
            
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            # Return diagonal matrix with minimum variance
            n = len(returns.columns)
            return pd.DataFrame(
                np.eye(n) * self.min_variance,
                index=returns.columns,
                columns=returns.columns
            )
