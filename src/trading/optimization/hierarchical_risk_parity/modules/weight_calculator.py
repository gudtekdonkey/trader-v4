"""
Weight calculation module for Hierarchical Risk Parity optimization.
Implements recursive bisection for HRP weight allocation.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class HRPWeightCalculator:
    """Calculates HRP weights using recursive bisection"""
    
    def __init__(self, min_variance: float = 1e-8):
        self.min_variance = min_variance
        
    def calculate_hrp_weights(self, covariance_matrix: pd.DataFrame, 
                            sorted_indices: List[int]) -> np.ndarray:
        """
        Calculate HRP weights using recursive bisection.
        
        Args:
            covariance_matrix: Asset covariance matrix
            sorted_indices: Quasi-diagonalized asset order
            
        Returns:
            Array of portfolio weights in original order
        """
        try:
            n_assets = len(sorted_indices)
            
            # Handle single asset
            if n_assets == 1:
                return np.array([1.0])
            
            weights = np.ones(n_assets)
            
            # Validate covariance matrix
            if covariance_matrix.shape != (n_assets, n_assets):
                logger.error("Covariance matrix dimension mismatch")
                return np.ones(n_assets) / n_assets
            
            # Reorder covariance matrix
            try:
                reordered_cov = covariance_matrix.iloc[sorted_indices, sorted_indices]
            except Exception as e:
                logger.error(f"Error reordering covariance matrix: {e}")
                return np.ones(n_assets) / n_assets
            
            # Recursive bisection
            cluster_weights = self._recursive_bisection(reordered_cov, list(range(n_assets)))
            
            # Validate cluster weights
            if len(cluster_weights) != n_assets:
                logger.error("Invalid cluster weights length")
                return np.ones(n_assets) / n_assets
            
            # Map back to original order
            original_weights = np.zeros(n_assets)
            for i, original_idx in enumerate(sorted_indices):
                if 0 <= original_idx < n_assets and i < len(cluster_weights):
                    original_weights[original_idx] = cluster_weights[i]
            
            # Ensure weights sum to 1
            weight_sum = original_weights.sum()
            if weight_sum > 0:
                original_weights = original_weights / weight_sum
            else:
                original_weights = np.ones(n_assets) / n_assets
            
            return original_weights
            
        except Exception as e:
            logger.error(f"Error calculating HRP weights: {e}")
            # Return equal weights as fallback
            n_assets = len(sorted_indices)
            return np.ones(n_assets) / n_assets
    
    def _recursive_bisection(self, covariance: pd.DataFrame, 
                           assets_indices: List[int]) -> np.ndarray:
        """
        Recursively calculate weights for asset cluster using bisection.
        
        Args:
            covariance: Covariance matrix (reordered)
            assets_indices: Indices of assets in current cluster
            
        Returns:
            Array of weights for the cluster
        """
        # Prevent infinite recursion
        if len(assets_indices) > covariance.shape[0]:
            logger.error("Invalid cluster size")
            return np.ones(len(assets_indices)) / len(assets_indices)
        
        if len(assets_indices) == 0:
            return np.array([])
            
        if len(assets_indices) == 1:
            return np.array([1.0])
        
        # Split cluster in half
        mid = len(assets_indices) // 2
        left_cluster = assets_indices[:mid]
        right_cluster = assets_indices[mid:]
        
        # Ensure non-empty clusters
        if not left_cluster or not right_cluster:
            return np.ones(len(assets_indices)) / len(assets_indices)
        
        try:
            # Calculate cluster variances
            left_cov = covariance.iloc[left_cluster, left_cluster]
            right_cov = covariance.iloc[right_cluster, right_cluster]
            
            # Calculate inverse variance weights
            left_var = self._calculate_cluster_variance(left_cov)
            right_var = self._calculate_cluster_variance(right_cov)
            
            left_ivp = 1.0 / np.sqrt(max(left_var, self.min_variance))
            right_ivp = 1.0 / np.sqrt(max(right_var, self.min_variance))
            
            # Allocate weight between clusters (inverse volatility)
            total_ivp = left_ivp + right_ivp
            if total_ivp > 0:
                left_weight = left_ivp / total_ivp
                right_weight = right_ivp / total_ivp
            else:
                left_weight = 0.5
                right_weight = 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating cluster weights: {e}")
            left_weight = 0.5
            right_weight = 0.5
        
        # Recursively get weights for sub-clusters
        left_weights = self._recursive_bisection(covariance, left_cluster) * left_weight
        right_weights = self._recursive_bisection(covariance, right_cluster) * right_weight
        
        return np.concatenate([left_weights, right_weights])
    
    def _calculate_cluster_variance(self, cluster_cov: pd.DataFrame) -> float:
        """
        Calculate variance for a cluster of assets.
        
        Args:
            cluster_cov: Covariance matrix for the cluster
            
        Returns:
            Cluster variance (using average variance approach)
        """
        try:
            # Use average variance of assets in cluster
            # This is a simplification - could use minimum variance portfolio
            cluster_var = np.diag(cluster_cov).mean()
            
            # Ensure positive variance
            return max(cluster_var, self.min_variance)
            
        except Exception as e:
            logger.error(f"Error calculating cluster variance: {e}")
            return self.min_variance
    
    def calculate_inverse_variance_weights(self, covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate simple inverse variance weights for comparison.
        
        Args:
            covariance_matrix: Asset covariance matrix
            
        Returns:
            Dictionary of asset weights
        """
        try:
            # Get variances (diagonal of covariance matrix)
            variances = np.diag(covariance_matrix.values)
            
            # Ensure positive variances
            variances = np.maximum(variances, self.min_variance)
            
            # Calculate inverse variance weights
            inv_var = 1.0 / variances
            weights = inv_var / inv_var.sum()
            
            # Create weight dictionary
            weight_dict = {}
            for i, asset in enumerate(covariance_matrix.index):
                weight_dict[asset] = weights[i]
            
            return weight_dict
            
        except Exception as e:
            logger.error(f"Error calculating inverse variance weights: {e}")
            # Return equal weights
            n_assets = len(covariance_matrix)
            return {asset: 1.0/n_assets for asset in covariance_matrix.index}
    
    def validate_and_normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Validate and normalize portfolio weights.
        
        Args:
            weights: Array of portfolio weights
            
        Returns:
            Normalized weight array
        """
        try:
            # Check for NaN or infinite values
            if not np.all(np.isfinite(weights)):
                logger.warning("Non-finite weights detected")
                weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
            
            # Ensure non-negative weights
            weights = np.maximum(weights, 0)
            
            # Handle all-zero weights
            if np.sum(weights) == 0:
                logger.warning("All weights are zero, using equal weights")
                n = len(weights)
                return np.ones(n) / n
            
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            
            # Final validation
            if not np.isclose(np.sum(weights), 1.0):
                logger.warning(f"Weight sum {np.sum(weights)} != 1.0, renormalizing")
                weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error validating weights: {e}")
            n = len(weights)
            return np.ones(n) / n
    
    def apply_weight_constraints(self, weights: np.ndarray, 
                               min_weight: float = 0.0,
                               max_weight: float = 1.0,
                               asset_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply min/max weight constraints to portfolio.
        
        Args:
            weights: Array of portfolio weights
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            asset_names: Optional list of asset names for logging
            
        Returns:
            Constrained weight array
        """
        try:
            constrained_weights = weights.copy()
            
            # Apply constraints
            constrained_weights = np.maximum(constrained_weights, min_weight)
            constrained_weights = np.minimum(constrained_weights, max_weight)
            
            # Log assets hitting constraints
            if asset_names:
                for i, (w, cw) in enumerate(zip(weights, constrained_weights)):
                    if w != cw:
                        asset = asset_names[i] if i < len(asset_names) else f"Asset_{i}"
                        if cw == min_weight:
                            logger.info(f"{asset} weight constrained to minimum {min_weight:.2%}")
                        elif cw == max_weight:
                            logger.info(f"{asset} weight constrained to maximum {max_weight:.2%}")
            
            # Renormalize
            return self.validate_and_normalize_weights(constrained_weights)
            
        except Exception as e:
            logger.error(f"Error applying weight constraints: {e}")
            return weights
