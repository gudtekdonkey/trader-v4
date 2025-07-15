"""
Hierarchical Risk Parity (HRP) Portfolio Optimization
Implementation of the HRP algorithm for portfolio construction
"""

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import logging
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio optimization algorithm
    Based on the work by Marcos López de Prado
    """
    
    def __init__(self, method='single', distance_metric='correlation'):
        """
        Initialize HRP optimizer
        
        Args:
            method: Linkage method for hierarchical clustering
            distance_metric: Distance metric for correlation matrix
        """
        self.method = method
        self.distance_metric = distance_metric
        
    def optimize_portfolio(self, returns: pd.DataFrame) -> Dict:
        """
        Optimize portfolio using HRP algorithm
        
        Args:
            returns: DataFrame of asset returns with columns as assets
            
        Returns:
            Dictionary containing weights and portfolio metrics
        """
        try:
            logger.info(f"Starting HRP optimization for {len(returns.columns)} assets")
            
            # Step 1: Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Step 2: Get distance matrix
            distance_matrix = self._get_distance_matrix(correlation_matrix)
            
            # Step 3: Perform hierarchical clustering
            clusters = self._cluster_assets(distance_matrix)
            
            # Step 4: Get quasi-diagonalization order
            sorted_indices = self._get_quasi_diag_order(clusters)
            
            # Step 5: Calculate HRP weights
            weights = self._get_hrp_weights(correlation_matrix, sorted_indices)
            
            # Step 6: Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            # Convert to dict with asset names
            weights_dict = {asset: weight for asset, weight in zip(returns.columns, weights)}
            
            result = {
                'weights': weights_dict,
                'correlation_matrix': correlation_matrix,
                'distance_matrix': distance_matrix,
                'cluster_order': sorted_indices,
                'portfolio_metrics': portfolio_metrics
            }
            
            logger.info(f"HRP optimization completed. Weights: {weights_dict}")
            return result
            
        except Exception as e:
            logger.error(f"Error in HRP optimization: {e}")
            # Return equal weights as fallback
            n_assets = len(returns.columns)
            equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
            return {
                'weights': equal_weights,
                'error': str(e),
                'portfolio_metrics': {}
            }
    
    def _get_distance_matrix(self, correlation_matrix: pd.DataFrame) -> np.ndarray:
        """Convert correlation matrix to distance matrix"""
        try:
            if self.distance_metric == 'correlation':
                # Distance = sqrt(0.5 * (1 - correlation))
                distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
            elif self.distance_metric == 'angular':
                # Angular distance
                distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
            else:
                # Default to correlation distance
                distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
            
            # Ensure diagonal is zero
            np.fill_diagonal(distance_matrix.values, 0)
            
            return distance_matrix.values
            
        except Exception as e:
            logger.error(f"Error calculating distance matrix: {e}")
            # Return identity matrix as fallback
            n = len(correlation_matrix)
            return np.eye(n)
    
    def _cluster_assets(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering on distance matrix"""
        try:
            # Convert to condensed form for scipy
            condensed_distances = squareform(distance_matrix, checks=False)
            
            # Perform hierarchical clustering
            clusters = linkage(condensed_distances, method=self.method)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            # Return simple clustering as fallback
            n = distance_matrix.shape[0]
            clusters = np.zeros((n-1, 4))
            for i in range(n-1):
                clusters[i] = [i, i+1, 1, 2]
            return clusters
    
    def _get_quasi_diag_order(self, clusters: np.ndarray) -> list:
        """Get quasi-diagonalization order from clusters"""
        try:
            # Get the order of assets based on clustering
            n_assets = len(clusters) + 1
            order = []
            
            def _get_cluster_order(cluster_id):
                """Recursively get order from cluster tree"""
                if cluster_id < n_assets:
                    # Leaf node (original asset)
                    return [cluster_id]
                else:
                    # Internal node
                    cluster_idx = int(cluster_id - n_assets)
                    left_child = int(clusters[cluster_idx, 0])
                    right_child = int(clusters[cluster_idx, 1])
                    
                    # Recursively get order from children
                    left_order = _get_cluster_order(left_child)
                    right_order = _get_cluster_order(right_child)
                    
                    return left_order + right_order
            
            # Start from root (last cluster)
            root_id = len(clusters) - 1 + n_assets
            order = _get_cluster_order(root_id)
            
            return order
            
        except Exception as e:
            logger.error(f"Error getting quasi-diagonal order: {e}")
            # Return simple order as fallback
            return list(range(len(clusters) + 1))
    
    def _get_hrp_weights(self, correlation_matrix: pd.DataFrame, sorted_indices: list) -> np.ndarray:
        """Calculate HRP weights using recursive bisection"""
        try:
            n_assets = len(sorted_indices)
            weights = np.ones(n_assets)
            
            # Reorder correlation matrix
            reordered_corr = correlation_matrix.iloc[sorted_indices, sorted_indices]
            
            # Recursive bisection
            def _get_cluster_weights(assets_indices):
                """Recursively calculate weights for asset cluster"""
                if len(assets_indices) == 1:
                    return np.array([1.0])
                
                # Split cluster in half
                mid = len(assets_indices) // 2
                left_cluster = assets_indices[:mid]
                right_cluster = assets_indices[mid:]
                
                # Calculate cluster variances
                left_corr = reordered_corr.iloc[left_cluster, left_cluster]
                right_corr = reordered_corr.iloc[right_cluster, right_cluster]
                
                # Calculate inverse variance weights within clusters
                left_ivp = 1.0 / np.diag(left_corr).mean() if len(left_cluster) > 0 else 1.0
                right_ivp = 1.0 / np.diag(right_corr).mean() if len(right_cluster) > 0 else 1.0
                
                # Allocate weight between clusters (inverse volatility)
                total_ivp = left_ivp + right_ivp
                left_weight = left_ivp / total_ivp if total_ivp > 0 else 0.5
                right_weight = right_ivp / total_ivp if total_ivp > 0 else 0.5
                
                # Recursively get weights for sub-clusters
                left_weights = _get_cluster_weights(left_cluster) * left_weight
                right_weights = _get_cluster_weights(right_cluster) * right_weight
                
                return np.concatenate([left_weights, right_weights])
            
            # Calculate weights for all assets
            cluster_weights = _get_cluster_weights(list(range(n_assets)))
            
            # Map back to original order
            original_weights = np.zeros(n_assets)
            for i, original_idx in enumerate(sorted_indices):
                original_weights[original_idx] = cluster_weights[i]
            
            # Normalize weights to sum to 1
            original_weights = original_weights / original_weights.sum()
            
            return original_weights
            
        except Exception as e:
            logger.error(f"Error calculating HRP weights: {e}")
            # Return equal weights as fallback
            n_assets = len(sorted_indices)
            return np.ones(n_assets) / n_assets
    
    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Basic metrics
            total_return = (1 + portfolio_returns).prod() - 1
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            
            # Diversification metrics
            individual_volatilities = returns.std() * np.sqrt(252)
            weighted_avg_vol = (weights * individual_volatilities).sum()
            diversification_ratio = weighted_avg_vol / volatility if volatility > 0 else 1.0
            
            # Risk metrics
            var_95 = np.percentile(portfolio_returns, 5)
            
            # Concentration metrics
            concentration = (weights ** 2).sum()  # Herfindahl index
            effective_assets = 1 / concentration
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'diversification_ratio': diversification_ratio,
                'var_95': var_95,
                'concentration': concentration,
                'effective_assets': effective_assets,
                'max_weight': weights.max(),
                'min_weight': weights.min()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def get_cluster_allocation(self, returns: pd.DataFrame, n_clusters: int = 3) -> Dict:
        """Get asset allocation by clusters"""
        try:
            correlation_matrix = returns.corr()
            distance_matrix = self._get_distance_matrix(correlation_matrix)
            clusters = self._cluster_assets(distance_matrix)
            
            # Get cluster labels
            cluster_labels = fcluster(clusters, n_clusters, criterion='maxclust')
            
            # Group assets by cluster
            cluster_allocation = {}
            for i, asset in enumerate(returns.columns):
                cluster_id = cluster_labels[i]
                if cluster_id not in cluster_allocation:
                    cluster_allocation[cluster_id] = []
                cluster_allocation[cluster_id].append(asset)
            
            return cluster_allocation
            
        except Exception as e:
            logger.error(f"Error getting cluster allocation: {e}")
            return {}
    
    def backtest_hrp(self, returns: pd.DataFrame, rebalance_freq: str = 'M') -> pd.DataFrame:
        """Backtest HRP strategy with periodic rebalancing"""
        try:
            # Resample returns based on rebalancing frequency
            if rebalance_freq == 'M':
                rebalance_dates = returns.resample('M').last().index
            elif rebalance_freq == 'Q':
                rebalance_dates = returns.resample('Q').last().index
            else:  # Daily
                rebalance_dates = returns.index
            
            portfolio_returns = []
            current_weights = None
            
            for date in returns.index:
                if date in rebalance_dates or current_weights is None:
                    # Rebalance portfolio
                    lookback_returns = returns.loc[:date].tail(60)  # 60-day lookback
                    if len(lookback_returns) >= 30:  # Minimum data requirement
                        hrp_result = self.optimize_portfolio(lookback_returns)
                        current_weights = np.array([hrp_result['weights'].get(asset, 0) 
                                                  for asset in returns.columns])
                    else:
                        # Equal weights if insufficient data
                        current_weights = np.ones(len(returns.columns)) / len(returns.columns)
                
                # Calculate portfolio return for the day
                if current_weights is not None:
                    daily_return = (returns.loc[date] * current_weights).sum()
                    portfolio_returns.append(daily_return)
                else:
                    portfolio_returns.append(0)
            
            return pd.Series(portfolio_returns, index=returns.index)
            
        except Exception as e:
            logger.error(f"Error in HRP backtest: {e}")
            return pd.Series(index=returns.index, dtype=float)

# Utility functions for HRP analysis
def plot_dendrogram(hrp_optimizer, returns: pd.DataFrame, figsize=(12, 8)):
    """Plot hierarchical clustering dendrogram"""
    try:
        import matplotlib.pyplot as plt
        
        correlation_matrix = returns.corr()
        distance_matrix = hrp_optimizer._get_distance_matrix(correlation_matrix)
        clusters = hrp_optimizer._cluster_assets(distance_matrix)
        
        plt.figure(figsize=figsize)
        dendrogram(clusters, labels=returns.columns.tolist(), orientation='top')
        plt.title('Asset Clustering Dendrogram')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")
    except Exception as e:
        logger.error(f"Error plotting dendrogram: {e}")

def compare_allocations(returns: pd.DataFrame) -> pd.DataFrame:
    """Compare HRP with other allocation methods"""
    try:
        hrp = HierarchicalRiskParity()
        
        # HRP allocation
        hrp_result = hrp.optimize_portfolio(returns)
        hrp_weights = hrp_result['weights']
        
        # Equal weight allocation
        n_assets = len(returns.columns)
        equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
        
        # Inverse volatility allocation
        volatilities = returns.std()
        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
        inv_vol_weights = inv_vol_weights.to_dict()
        
        # Market cap weighted (using volume as proxy)
        volumes = returns.abs().mean()  # Use absolute returns as volume proxy
        market_cap_weights = volumes / volumes.sum()
        market_cap_weights = market_cap_weights.to_dict()
        
        # Combine results
        comparison_df = pd.DataFrame({
            'HRP': pd.Series(hrp_weights),
            'Equal_Weight': pd.Series(equal_weights),
            'Inverse_Volatility': pd.Series(inv_vol_weights),
            'Market_Cap_Proxy': pd.Series(market_cap_weights)
        }).fillna(0)
        
        return comparison_df
        
    except Exception as e:
        logger.error(f"Error comparing allocations: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Generate sample returns data
    np.random.seed(42)
    n_assets = 5
    n_days = 252
    
    # Create correlated returns
    correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    
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
    print(f"Weights: {result['weights']}")
    print(f"Portfolio Metrics: {result['portfolio_metrics']}")
    
    # Compare with other methods
    comparison = compare_allocations(returns)
    print("\nAllocation Comparison:")
    print(comparison)
