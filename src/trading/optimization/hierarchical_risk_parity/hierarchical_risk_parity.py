"""
Hierarchical Risk Parity Portfolio Optimizer - Main coordination module
Portfolio optimization using hierarchical clustering and risk parity.

File: hierarchical_risk_parity.py
Modified: 2024-12-19
Refactored: 2025-01-18
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
import warnings
import traceback
warnings.filterwarnings('ignore')

# Import modularized components
from .modules.data_preprocessor import DataPreprocessor
from .modules.clustering import HierarchicalClustering
from .modules.weight_calculator import HRPWeightCalculator
from .modules.portfolio_analytics import PortfolioAnalytics

logger = logging.getLogger(__name__)


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio optimization algorithm.
    Based on the work by Marcos López de Prado.
    
    This is the main coordination class that delegates to specialized modules:
    - data_preprocessor: Data cleaning and correlation calculations
    - clustering: Hierarchical clustering and quasi-diagonalization
    - weight_calculator: HRP weight calculation using recursive bisection
    - portfolio_analytics: Performance metrics and backtesting
    """
    
    def __init__(self, method: str = 'single', distance_metric: str = 'correlation'):
        """
        Initialize HRP optimizer.
        
        Args:
            method: Linkage method for hierarchical clustering
            distance_metric: Distance metric for correlation matrix
        """
        # Validate parameters
        valid_methods = ['single', 'complete', 'average', 'ward']
        if method not in valid_methods:
            logger.warning(f"Invalid method {method}, using 'single'")
            method = 'single'
            
        valid_metrics = ['correlation', 'angular']
        if distance_metric not in valid_metrics:
            logger.warning(f"Invalid distance metric {distance_metric}, using 'correlation'")
            distance_metric = 'correlation'
            
        self.method = method
        self.distance_metric = distance_metric
        
        # Numerical stability parameters
        self.min_variance = 1e-8
        self.max_iterations = 1000
        self.numerical_tolerance = 1e-10
        
        # Initialize module managers
        self.data_preprocessor = DataPreprocessor(
            min_variance=self.min_variance,
            numerical_tolerance=self.numerical_tolerance
        )
        self.clustering = HierarchicalClustering(method=method)
        self.weight_calculator = HRPWeightCalculator(min_variance=self.min_variance)
        self.analytics = PortfolioAnalytics(numerical_tolerance=self.numerical_tolerance)
        
    def optimize_portfolio(self, returns: pd.DataFrame) -> Dict:
        """
        Optimize portfolio using HRP algorithm.
        
        Args:
            returns: DataFrame of asset returns with columns as assets
            
        Returns:
            Dictionary containing weights and portfolio metrics
        """
        try:
            # Validate input
            is_valid, error_msg = self.data_preprocessor.validate_returns_data(returns)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Clean data
            returns_clean = self.data_preprocessor.clean_returns_data(returns)
            
            logger.info(f"Starting HRP optimization for {len(returns_clean.columns)} assets")
            
            # Step 1: Calculate correlation matrix
            correlation_matrix = self.data_preprocessor.calculate_correlation_matrix(returns_clean)
            
            # Step 2: Get distance matrix
            distance_matrix = self.data_preprocessor.calculate_distance_matrix(
                correlation_matrix, self.distance_metric
            )
            
            # Step 3: Perform hierarchical clustering
            clusters = self.clustering.cluster_assets(distance_matrix)
            
            # Step 4: Get quasi-diagonalization order
            sorted_indices = self.clustering.get_quasi_diag_order(
                clusters, len(returns_clean.columns)
            )
            
            # Validate indices
            if not self.clustering.validate_indices(sorted_indices, len(returns_clean.columns)):
                logger.error("Invalid sorted indices")
                raise ValueError("Clustering produced invalid indices")
            
            # Step 5: Calculate covariance matrix for weight calculation
            covariance_matrix = self.data_preprocessor.calculate_covariance_matrix(
                returns_clean, correlation_matrix
            )
            
            # Step 6: Calculate HRP weights
            weights = self.weight_calculator.calculate_hrp_weights(
                covariance_matrix, sorted_indices
            )
            
            # Validate weights
            weights = self.weight_calculator.validate_and_normalize_weights(weights)
            
            # Step 7: Calculate portfolio metrics
            portfolio_metrics = self.analytics.calculate_portfolio_metrics(
                returns_clean, weights
            )
            
            # Convert to dict with asset names
            weights_dict = {asset: weight for asset, weight in zip(returns_clean.columns, weights)}
            
            result = {
                'weights': weights_dict,
                'correlation_matrix': correlation_matrix,
                'distance_matrix': distance_matrix,
                'cluster_order': sorted_indices,
                'portfolio_metrics': portfolio_metrics,
                'status': 'success'
            }
            
            logger.info(f"HRP optimization completed. Max weight: {max(weights_dict.values()):.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Error in HRP optimization: {e}")
            logger.error(traceback.format_exc())
            
            # Return equal weights as fallback
            n_assets = len(returns.columns)
            equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
            
            return {
                'weights': equal_weights,
                'error': str(e),
                'portfolio_metrics': self._calculate_fallback_metrics(returns),
                'status': 'fallback'
            }
    
    def _calculate_fallback_metrics(self, returns: pd.DataFrame) -> Dict:
        """Calculate basic metrics for fallback case"""
        try:
            # Equal weight portfolio
            n_assets = len(returns.columns)
            equal_weights = np.ones(n_assets) / n_assets
            
            portfolio_returns = (returns * equal_weights).sum(axis=1)
            
            return {
                'total_return': (1 + portfolio_returns).prod() - 1,
                'volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': 0,
                'note': 'Fallback metrics with equal weights'
            }
        except Exception as e:
            logger.error(f"Error in fallback metrics: {e}")
            return {
                'error': str(e),
                'total_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0
            }
    
    def get_cluster_allocation(self, returns: pd.DataFrame, n_clusters: int = 3) -> Dict:
        """
        Get asset allocation by clusters.
        
        Args:
            returns: DataFrame of asset returns
            n_clusters: Desired number of clusters
            
        Returns:
            Dictionary mapping cluster IDs to asset lists
        """
        try:
            # Validate inputs
            is_valid, error_msg = self.data_preprocessor.validate_returns_data(returns)
            if not is_valid:
                raise ValueError(error_msg)
                
            n_clusters = max(1, min(n_clusters, len(returns.columns)))
            
            # Calculate correlation and distance matrices
            correlation_matrix = self.data_preprocessor.calculate_correlation_matrix(returns)
            distance_matrix = self.data_preprocessor.calculate_distance_matrix(
                correlation_matrix, self.distance_metric
            )
            
            # Perform clustering
            clusters = self.clustering.cluster_assets(distance_matrix)
            
            # Get cluster allocation
            cluster_allocation = self.clustering.get_cluster_allocation(
                list(returns.columns), clusters, n_clusters
            )
            
            return cluster_allocation
            
        except Exception as e:
            logger.error(f"Error getting cluster allocation: {e}")
            # Return all assets in one cluster as fallback
            return {1: list(returns.columns)}
    
    def backtest_hrp(self, returns: pd.DataFrame, rebalance_freq: str = 'M') -> pd.Series:
        """
        Backtest HRP strategy with periodic rebalancing.
        
        Args:
            returns: DataFrame of asset returns
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Series of HRP strategy returns
        """
        def hrp_weight_function(lookback_returns):
            """Weight calculation function for backtesting"""
            try:
                result = self.optimize_portfolio(lookback_returns)
                if result.get('status') == 'success':
                    return result['weights']
                else:
                    # Equal weights if optimization fails
                    n = len(lookback_returns.columns)
                    return {asset: 1.0/n for asset in lookback_returns.columns}
            except Exception:
                n = len(lookback_returns.columns)
                return {asset: 1.0/n for asset in lookback_returns.columns}
        
        return self.analytics.backtest_strategy(
            returns, hrp_weight_function, rebalance_freq
        )
    
    def compare_allocations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compare HRP with other allocation methods.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            DataFrame comparing different allocation methods
        """
        try:
            # Get HRP weights
            hrp_result = self.optimize_portfolio(returns)
            hrp_weights = hrp_result.get('weights', {})
            
            # Compare with other methods
            return self.analytics.compare_allocations(returns, hrp_weights)
            
        except Exception as e:
            logger.error(f"Error comparing allocations: {e}")
            return pd.DataFrame()
    
    def get_inverse_variance_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate inverse variance weights for comparison.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary of asset weights
        """
        try:
            covariance_matrix = self.data_preprocessor.calculate_covariance_matrix(returns)
            return self.weight_calculator.calculate_inverse_variance_weights(covariance_matrix)
        except Exception as e:
            logger.error(f"Error calculating inverse variance weights: {e}")
            n_assets = len(returns.columns)
            return {asset: 1.0/n_assets for asset in returns.columns}


# Utility functions for HRP analysis
def plot_dendrogram(hrp_optimizer: HierarchicalRiskParity, 
                   returns: pd.DataFrame, 
                   figsize: Tuple[int, int] = (12, 8)):
    """Plot hierarchical clustering dendrogram"""
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram
        
        # Validate inputs
        if returns is None or returns.empty:
            logger.error("Cannot plot dendrogram with empty returns")
            return
        
        # Calculate matrices
        correlation_matrix = hrp_optimizer.data_preprocessor.calculate_correlation_matrix(returns)
        distance_matrix = hrp_optimizer.data_preprocessor.calculate_distance_matrix(
            correlation_matrix, hrp_optimizer.distance_metric
        )
        clusters = hrp_optimizer.clustering.cluster_assets(distance_matrix)
        
        if clusters.size == 0:
            logger.warning("No clusters to plot")
            return
        
        plt.figure(figsize=figsize)
        dendrogram(clusters, labels=returns.columns.tolist(), orientation='top')
        plt.title('Asset Clustering Dendrogram')
        plt.xlabel('Assets')
        plt.ylabel('Distance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")
    except Exception as e:
        logger.error(f"Error plotting dendrogram: {e}")


def compare_allocations(returns: pd.DataFrame) -> pd.DataFrame:
    """Compare HRP with other allocation methods using default parameters"""
    try:
        if returns is None or returns.empty:
            raise ValueError("Returns data is empty")
            
        hrp = HierarchicalRiskParity()
        return hrp.compare_allocations(returns)
        
    except Exception as e:
        logger.error(f"Error comparing allocations: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    # Example usage with error handling
    import numpy as np
    
    try:
        # Generate sample returns data
        np.random.seed(42)
        n_assets = 5
        n_days = 252
        
        # Create correlated returns
        correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1.0)
        
        # Ensure positive definite correlation matrix
        eigenvalues = np.linalg.eigvals(correlation)
        if np.min(eigenvalues) < 0:
            correlation = correlation + (-np.min(eigenvalues) + 0.01) * np.eye(n_assets)
        
        returns = pd.DataFrame(
            np.random.multivariate_normal(
                mean=[0.001] * n_assets,
                cov=correlation * 0.02**2,  # 2% daily volatility
                size=n_days
            ),
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Test HRP optimization
        hrp = HierarchicalRiskParity()
        result = hrp.optimize_portfolio(returns)
        
        print("HRP Optimization Results:")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Weights: {result['weights']}")
        print(f"Portfolio Metrics: {result.get('portfolio_metrics', {})}")
        
        # Compare with other methods
        comparison = hrp.compare_allocations(returns)
        print("\nAllocation Comparison:")
        print(comparison)
        
        # Test backtesting
        backtest_results = hrp.backtest_hrp(returns, rebalance_freq='M')
        print(f"\nBacktest cumulative return: {(1 + backtest_results).prod() - 1:.2%}")
        
    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()


"""
MODULARIZATION SUMMARY:
- Original file: 1,200+ lines
- Refactored main file: ~350 lines
- Modules created:
  1. data_preprocessor.py (~300 lines) - Data cleaning and correlation calculations
  2. clustering.py (~350 lines) - Hierarchical clustering algorithms
  3. weight_calculator.py (~350 lines) - HRP weight calculation
  4. portfolio_analytics.py (~400 lines) - Portfolio metrics and backtesting
  
Benefits:
- Clear separation of data processing, clustering, and weight calculation
- Isolated portfolio analytics
- Reusable clustering algorithms
- Better testability of mathematical operations
- Easier to extend with new distance metrics or clustering methods
"""
