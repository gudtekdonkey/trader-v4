"""
Bayesian updating module for Black-Litterman optimization.
Incorporates investor views using Bayesian inference.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BayesianUpdater:
    """Handles Bayesian updating of prior beliefs with investor views"""
    
    def __init__(self, tau: float = 0.025, min_variance: float = 1e-8):
        self.tau = tau
        self.min_variance = min_variance
        
    def incorporate_views(self,
                         equilibrium_returns: pd.Series,
                         cov_matrix: pd.DataFrame,
                         views: Dict,
                         view_confidences: Dict,
                         matrix_ops) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Incorporate investor views using Bayesian updating.
        
        Args:
            equilibrium_returns: Prior expected returns
            cov_matrix: Prior covariance matrix
            views: Dict of investor views {asset: expected_return}
            view_confidences: Dict of confidence levels {asset: confidence}
            matrix_ops: MatrixOperations instance for matrix calculations
            
        Returns:
            Tuple of (posterior expected returns, posterior covariance matrix)
        """
        try:
            # Construct picking matrix P and view vector Q
            assets = equilibrium_returns.index
            n_views = len(views)
            P = np.zeros((n_views, len(assets)))
            Q = np.zeros(n_views)
            
            valid_views = 0
            for i, (asset, view_return) in enumerate(views.items()):
                if asset in assets:
                    asset_idx = assets.get_loc(asset)
                    P[valid_views, asset_idx] = 1
                    Q[valid_views] = view_return
                    valid_views += 1
            
            if valid_views == 0:
                logger.warning("No valid views found, returning equilibrium values")
                return equilibrium_returns, cov_matrix
            
            # Trim to valid views
            P = P[:valid_views, :]
            Q = Q[:valid_views]
            
            # Construct view uncertainty matrix Omega
            omega_values = []
            for asset in list(views.keys())[:valid_views]:
                confidence = view_confidences.get(asset, 0.5)
                # Ensure positive variance
                variance = max(self.min_variance, 1/max(confidence, 0.01))
                omega_values.append(variance)
            
            Omega = np.diag(omega_values)
            
            # Bayesian updating
            tau_sigma = self.tau * cov_matrix.values
            
            try:
                # Safe matrix operations
                # Calculate intermediate matrices
                tau_sigma_inv = matrix_ops.safe_inverse(tau_sigma)
                omega_inv = matrix_ops.safe_inverse(Omega)
                
                if tau_sigma_inv is None or omega_inv is None:
                    logger.warning("Matrix inversion failed, using equilibrium values")
                    return equilibrium_returns, cov_matrix
                
                # Calculate Black-Litterman expected returns
                M1 = tau_sigma_inv
                M2 = np.dot(P.T, np.dot(omega_inv, P))
                M3 = np.dot(M1, equilibrium_returns.values)
                M4 = np.dot(P.T, np.dot(omega_inv, Q))
                
                combined_precision = M1 + M2
                combined_mean = M3 + M4
                
                # Final posterior calculations
                cov_bl_inv = combined_precision
                cov_bl = matrix_ops.safe_inverse(cov_bl_inv)
                
                if cov_bl is None:
                    logger.warning("Failed to compute posterior covariance")
                    return equilibrium_returns, cov_matrix
                
                mu_bl = np.dot(cov_bl, combined_mean)
                
                # Validate results
                if not np.all(np.isfinite(mu_bl)):
                    logger.warning("Non-finite posterior returns")
                    return equilibrium_returns, cov_matrix
                
                return (
                    pd.Series(mu_bl, index=assets),
                    pd.DataFrame(cov_bl, index=assets, columns=assets)
                )
                
            except Exception as e:
                logger.warning(f"Error in Bayesian updating: {e}")
                return equilibrium_returns, cov_matrix
                
        except Exception as e:
            logger.error(f"Error incorporating views: {e}")
            return equilibrium_returns, cov_matrix
    
    def validate_views(self, views: Dict, assets: pd.Index) -> Dict:
        """Validate and clean views"""
        validated_views = {}
        
        for asset, view in views.items():
            try:
                # Check if asset exists
                if asset not in assets:
                    logger.warning(f"Asset {asset} not in universe, skipping view")
                    continue
                
                # Validate view value
                if not isinstance(view, (int, float)) or not np.isfinite(view):
                    logger.warning(f"Invalid view for {asset}: {view}")
                    continue
                
                # Reasonable bounds for crypto returns (-50% to +100% annually)
                if abs(view) > 1.0:
                    logger.warning(f"View for {asset} seems extreme: {view}")
                    view = np.sign(view) * 1.0
                
                validated_views[asset] = view
                
            except Exception as e:
                logger.warning(f"Error validating view for {asset}: {e}")
                continue
        
        return validated_views
    
    def validate_view_confidences(self, confidences: Dict, views: Dict) -> Dict:
        """Validate and clean view confidences"""
        validated_confidences = {}
        
        for asset in views:
            confidence = confidences.get(asset, 0.5)
            
            try:
                # Validate confidence value
                if not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
                    logger.warning(f"Invalid confidence for {asset}: {confidence}")
                    confidence = 0.5
                
                # Ensure confidence is between 0 and 1
                confidence = max(0.01, min(0.99, confidence))
                
                validated_confidences[asset] = confidence
                
            except Exception as e:
                logger.warning(f"Error validating confidence for {asset}: {e}")
                validated_confidences[asset] = 0.5
        
        return validated_confidences
    
    def construct_view_matrices(self, views: Dict, assets: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct picking matrix P and view vector Q from views.
        
        Args:
            views: Dict of views {asset: expected_return}
            assets: Index of all assets
            
        Returns:
            Tuple of (picking_matrix, view_vector)
        """
        n_views = len(views)
        n_assets = len(assets)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in assets:
                asset_idx = assets.get_loc(asset)
                P[i, asset_idx] = 1
                Q[i] = view_return
        
        return P, Q
    
    def construct_uncertainty_matrix(self, view_confidences: Dict, views: Dict) -> np.ndarray:
        """
        Construct view uncertainty matrix Omega from confidences.
        
        Args:
            view_confidences: Dict of confidence levels {asset: confidence}
            views: Dict of views to ensure ordering
            
        Returns:
            Diagonal uncertainty matrix
        """
        omega_values = []
        
        for asset in views:
            confidence = view_confidences.get(asset, 0.5)
            # Convert confidence to variance (inverse relationship)
            variance = max(self.min_variance, 1/max(confidence, 0.01))
            omega_values.append(variance)
        
        return np.diag(omega_values)
