"""
Black-Litterman Portfolio Optimization for Crypto Assets

This module implements an advanced Black-Litterman portfolio optimization framework
specifically designed for cryptocurrency assets, including view generation from
multiple sources (ML models, technical analysis, and sentiment data).
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class BlackLittermanOptimizer:
    """
    Advanced Black-Litterman portfolio optimization for crypto assets.
    
    The Black-Litterman model combines market equilibrium with investor views
    to produce optimal portfolio weights. This implementation includes:
    - Covariance matrix shrinkage (Ledoit-Wolf)
    - Multiple constraint types
    - Fallback mechanisms for numerical stability
    
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
        self.risk_aversion = risk_aversion
        self.tau = tau
    
    def optimize_portfolio(
        self,
        returns_data: pd.DataFrame,
        market_caps: pd.Series,
        views: Optional[Dict] = None,
        view_confidences: Optional[Dict] = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
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
            # Step 1: Calculate market portfolio weights (equilibrium)
            market_weights = self._calculate_market_weights(market_caps)
            
            # Step 2: Calculate covariance matrix with shrinkage
            cov_matrix = self._calculate_covariance_matrix(
                returns_data, method='shrinkage'
            )
            
            # Step 3: Calculate implied equilibrium returns
            equilibrium_returns = self._calculate_equilibrium_returns(
                market_weights, cov_matrix
            )
            
            # Step 4: Incorporate investor views
            if views is not None and view_confidences is not None:
                bl_returns, bl_cov = self._incorporate_views(
                    equilibrium_returns, cov_matrix, views, view_confidences
                )
            else:
                bl_returns = equilibrium_returns
                bl_cov = cov_matrix
            
            # Step 5: Optimize portfolio with constraints
            optimal_weights = self._optimize_weights(
                bl_returns, bl_cov, constraints
            )
            
            # Calculate portfolio metrics
            expected_return = np.dot(optimal_weights, bl_returns)
            portfolio_volatility = np.sqrt(
                np.dot(optimal_weights, np.dot(bl_cov, optimal_weights))
            )
            sharpe_ratio = (
                expected_return / portfolio_volatility 
                if portfolio_volatility > 0 else 0
            )
            
            return {
                'weights': optimal_weights,
                'expected_returns': bl_returns,
                'covariance': bl_cov,
                'expected_portfolio_return': expected_return,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'market_weights': market_weights,
                'equilibrium_returns': equilibrium_returns
            }
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            # Fallback to equal weights
            n_assets = len(returns_data.columns)
            fallback_weights = pd.Series(
                np.ones(n_assets) / n_assets,
                index=returns_data.columns
            )
            return {
                'weights': fallback_weights,
                'expected_returns': returns_data.mean(),
                'covariance': returns_data.cov(),
                'expected_portfolio_return': returns_data.mean().mean(),
                'portfolio_volatility': 0.02,
                'sharpe_ratio': 0,
                'status': 'fallback'
            }
    
    def _calculate_market_weights(self, market_caps: pd.Series) -> pd.Series:
        """
        Calculate market capitalization weights.
        
        Args:
            market_caps: Series of market capitalizations
            
        Returns:
            Series of normalized market weights
        """
        total_market_cap = market_caps.sum()
        if total_market_cap > 0:
            return market_caps / total_market_cap
        else:
            # Equal weights fallback
            return pd.Series(
                np.ones(len(market_caps)) / len(market_caps),
                index=market_caps.index
            )
    
    def _calculate_covariance_matrix(
        self,
        returns_data: pd.DataFrame,
        method: str = 'shrinkage'
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix with various methods.
        
        Args:
            returns_data: DataFrame of asset returns
            method: Covariance estimation method ('shrinkage', 'exponential', 'simple')
            
        Returns:
            Covariance matrix as DataFrame
        """
        if method == 'exponential':
            # Exponentially weighted covariance
            return returns_data.ewm(span=60).cov().iloc[-len(returns_data.columns):]
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage
            return self._ledoit_wolf_shrinkage(returns_data)
        else:
            # Simple sample covariance
            return returns_data.cov()
    
    def _ledoit_wolf_shrinkage(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator for covariance matrix.
        
        This method shrinks the sample covariance matrix towards a structured
        estimator (scaled identity matrix) to improve out-of-sample performance.
        
        Args:
            returns_data: DataFrame of asset returns
            
        Returns:
            Shrunk covariance matrix as DataFrame
        """
        T, N = returns_data.shape
        
        if T < N:
            logger.warning("Not enough observations for reliable covariance estimation")
            return returns_data.cov()
        
        # Sample covariance matrix
        S = returns_data.cov().values
        
        # Shrinkage target (identity matrix scaled by average variance)
        mu = np.trace(S) / N
        F = mu * np.eye(N)
        
        # Calculate shrinkage intensity
        X = returns_data.values
        X_centered = X - X.mean(axis=0)
        
        # Shrinkage intensity calculation (simplified)
        pi_hat = np.sum((X_centered**2).T @ (X_centered**2)) / T
        rho_hat = np.sum(np.diag((X_centered.T @ X_centered / T - S)**2))
        gamma_hat = np.linalg.norm(S - F, 'fro')**2
        
        kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 1
        shrinkage = max(0, min(1, kappa / T))
        
        # Shrunk covariance matrix
        shrunk_cov = shrinkage * F + (1 - shrinkage) * S
        
        return pd.DataFrame(
            shrunk_cov,
            index=returns_data.columns,
            columns=returns_data.columns
        )
    
    def _calculate_equilibrium_returns(
        self,
        market_weights: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate implied equilibrium returns using reverse optimization.
        
        Args:
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix
            
        Returns:
            Series of equilibrium expected returns
        """
        # Reverse optimization: mu = lambda * Sigma * w
        return self.risk_aversion * np.dot(cov_matrix, market_weights)
    
    def _incorporate_views(
        self,
        equilibrium_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        views: Dict,
        view_confidences: Dict
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Incorporate investor views using Bayesian updating.
        
        Args:
            equilibrium_returns: Prior expected returns
            cov_matrix: Prior covariance matrix
            views: Dict of investor views {asset: expected_return}
            view_confidences: Dict of confidence levels {asset: confidence}
            
        Returns:
            Tuple of (posterior expected returns, posterior covariance matrix)
        """
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
        Omega = np.diag([
            1/view_confidences.get(asset, 1.0)
            for asset in list(views.keys())[:valid_views]
        ])
        
        # Bayesian updating
        tau_sigma = self.tau * cov_matrix.values
        
        try:
            # Calculate Black-Litterman expected returns
            M1 = linalg.inv(tau_sigma)
            M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
            M3 = np.dot(M1, equilibrium_returns.values)
            M4 = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
            
            mu_bl = np.dot(linalg.inv(M1 + M2), M3 + M4)
            
            # Calculate Black-Litterman covariance matrix
            cov_bl = linalg.inv(M1 + M2)
            
            return (
                pd.Series(mu_bl, index=assets),
                pd.DataFrame(cov_bl, index=assets, columns=assets)
            )
            
        except linalg.LinAlgError as e:
            logger.warning(f"Matrix inversion failed: {e}. Using equilibrium values.")
            return equilibrium_returns, cov_matrix
    
    def _optimize_weights(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> pd.Series:
        """
        Optimize portfolio weights with constraints.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            constraints: Portfolio constraints dict
            
        Returns:
            Series of optimal portfolio weights
        """
        n_assets = len(expected_returns)
        
        # Default constraints for crypto
        default_constraints = {
            'max_weight': 0.4,  # Max 40% per asset
            'min_weight': 0.0,  # No short selling
            'max_concentration': 0.6,  # Max 60% in top 3 assets
            'min_diversification': 3  # Minimum 3 assets
        }
        
        if constraints:
            default_constraints.update(constraints)
        
        # Objective function: maximize utility
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance)
        
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Concentration constraint
        if 'max_concentration' in default_constraints:
            max_conc = default_constraints['max_concentration']
            
            def concentration_constraint(weights):
                sorted_weights = np.sort(weights)[::-1]  # Descending order
                top_3_weight = np.sum(sorted_weights[:3])
                return max_conc - top_3_weight
            
            constraint_list.append({
                'type': 'ineq',
                'fun': concentration_constraint
            })
        
        # Minimum diversification
        if 'min_diversification' in default_constraints:
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
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list
            )
            
            if result.success:
                weights = result.x
                # Ensure weights are non-negative and sum to 1
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
                return pd.Series(weights, index=expected_returns.index)
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return self._fallback_weights(expected_returns, default_constraints)
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return self._fallback_weights(expected_returns, default_constraints)
    
    def _fallback_weights(
        self,
        expected_returns: pd.Series,
        constraints: Dict
    ) -> pd.Series:
        """
        Fallback weight calculation if optimization fails.
        
        Args:
            expected_returns: Expected returns for each asset
            constraints: Portfolio constraints dict
            
        Returns:
            Series of fallback portfolio weights
        """
        n_assets = len(expected_returns)
        max_weight = constraints.get('max_weight', 0.4)
        
        # Risk parity-like allocation
        weights = np.ones(n_assets) / n_assets
        
        # Cap maximum weights
        weights = np.minimum(weights, max_weight)
        weights = weights / np.sum(weights)
        
        return pd.Series(weights, index=expected_returns.index)


class CryptoViewGenerator:
    """
    Generate investor views for Black-Litterman optimization.
    
    This class provides methods to generate views from multiple sources:
    - Machine learning model predictions
    - Technical analysis indicators
    - Sentiment analysis data
    
    Attributes:
        ml_models: Dict of ML models for each asset
        market_data: Market data for generating views
    """
    
    def __init__(
        self,
        ml_models: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ):
        """
        Initialize the view generator.
        
        Args:
            ml_models: Dict of ML models {symbol: model}
            market_data: Dict of market data {symbol: DataFrame}
        """
        self.ml_models = ml_models or {}
        self.market_data = market_data
    
    def generate_ml_views(
        self,
        confidence_threshold: float = 0.6
    ) -> Tuple[Dict, Dict]:
        """
        Generate views based on ML predictions.
        
        Args:
            confidence_threshold: Minimum confidence level to include a view
            
        Returns:
            Tuple of (views dict, confidences dict)
        """
        views = {}
        view_confidences = {}
        
        for symbol, model in self.ml_models.items():
            try:
                # Get ML prediction
                if hasattr(model, 'predict') and self.market_data is not None:
                    prediction = model.predict(
                        self.market_data.get(symbol, pd.DataFrame())
                    )
                    
                    if isinstance(prediction, dict):
                        confidence = prediction.get('confidence', 0.5)
                        expected_return = prediction.get('expected_return', 0)
                        
                        if confidence >= confidence_threshold:
                            views[symbol] = expected_return
                            view_confidences[symbol] = confidence
                            
            except Exception as e:
                logger.warning(f"Error generating view for {symbol}: {e}")
                continue
        
        return views, view_confidences
    
    def generate_technical_views(
        self,
        technical_data: pd.DataFrame
    ) -> Tuple[Dict, Dict]:
        """
        Generate views based on technical analysis.
        
        Args:
            technical_data: DataFrame with technical indicators for each asset
            
        Returns:
            Tuple of (views dict, confidences dict)
        """
        views = {}
        view_confidences = {}
        
        for symbol in technical_data.columns:
            try:
                # Get technical indicators
                symbol_data = technical_data[symbol]
                
                # RSI-based view
                rsi = symbol_data.get('rsi', 50)
                if rsi < 30:  # Oversold
                    views[symbol] = 0.05  # Expect 5% return
                    view_confidences[symbol] = (30 - rsi) / 30
                elif rsi > 70:  # Overbought
                    views[symbol] = -0.03  # Expect -3% return
                    view_confidences[symbol] = (rsi - 70) / 30
                
                # MACD-based view enhancement
                macd = symbol_data.get('macd', 0)
                if symbol in views and macd != 0:
                    if ((views[symbol] > 0 and macd > 0) or
                        (views[symbol] < 0 and macd < 0)):
                        view_confidences[symbol] = min(
                            0.9,
                            view_confidences[symbol] * 1.2
                        )
                
            except Exception as e:
                logger.warning(f"Error generating technical view for {symbol}: {e}")
                continue
        
        return views, view_confidences
    
    def generate_sentiment_views(
        self,
        sentiment_data: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Generate views based on sentiment data.
        
        Args:
            sentiment_data: Dict of sentiment scores {symbol: score}
                           Scores should be between -1 and 1
            
        Returns:
            Tuple of (views dict, confidences dict)
        """
        views = {}
        view_confidences = {}
        
        for symbol, sentiment_score in sentiment_data.items():
            try:
                # Convert sentiment to expected return
                if abs(sentiment_score) > 0.3:  # Only use strong sentiment
                    expected_return = sentiment_score * 0.08  # Max 8% expected return
                    confidence = abs(sentiment_score)
                    
                    views[symbol] = expected_return
                    view_confidences[symbol] = confidence
                    
            except Exception as e:
                logger.warning(f"Error generating sentiment view for {symbol}: {e}")
                continue
        
        return views, view_confidences
    
    def combine_views(self, *view_sets) -> Tuple[Dict, Dict]:
        """
        Combine multiple sets of views using confidence-weighted averaging.
        
        Args:
            *view_sets: Variable number of (views, confidences) tuples
            
        Returns:
            Combined (views dict, confidences dict)
        """
        combined_views = {}
        combined_confidences = {}
        
        # Collect all views for each symbol
        symbol_views = {}
        symbol_confidences = {}
        
        for views, confidences in view_sets:
            for symbol, view in views.items():
                if symbol not in symbol_views:
                    symbol_views[symbol] = []
                    symbol_confidences[symbol] = []
                
                symbol_views[symbol].append(view)
                symbol_confidences[symbol].append(confidences[symbol])
        
        # Combine views using confidence-weighted averaging
        for symbol in symbol_views:
            views_list = symbol_views[symbol]
            conf_list = symbol_confidences[symbol]
            
            if views_list:
                # Weighted average of views
                total_conf = sum(conf_list)
                if total_conf > 0:
                    weighted_view = sum(
                        v * c for v, c in zip(views_list, conf_list)
                    ) / total_conf
                    avg_confidence = total_conf / len(views_list)
                    
                    combined_views[symbol] = weighted_view
                    combined_confidences[symbol] = min(0.95, avg_confidence)
        
        return combined_views, combined_confidences