"""
Clustering module for Hierarchical Risk Parity optimization.
Handles hierarchical clustering and quasi-diagonalization.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class HierarchicalClustering:
    """Handles clustering operations for HRP optimization"""
    
    def __init__(self, method: str = 'single'):
        """
        Initialize clustering module.
        
        Args:
            method: Linkage method for hierarchical clustering
        """
        # Validate method
        valid_methods = ['single', 'complete', 'average', 'ward']
        if method not in valid_methods:
            logger.warning(f"Invalid method {method}, using 'single'")
            method = 'single'
            
        self.method = method
        
    def cluster_assets(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Perform hierarchical clustering on distance matrix.
        
        Args:
            distance_matrix: Square distance matrix
            
        Returns:
            Linkage matrix from scipy hierarchical clustering
        """
        try:
            # Validate distance matrix
            if distance_matrix.size == 0:
                raise ValueError("Empty distance matrix")
            
            # Convert to condensed form for scipy
            # Ensure matrix is square
            n = distance_matrix.shape[0]
            if distance_matrix.shape != (n, n):
                raise ValueError(f"Distance matrix must be square, got shape {distance_matrix.shape}")
            
            # Handle single asset case
            if n == 1:
                return np.array([])
            
            # Ensure valid condensed form
            try:
                condensed_distances = squareform(distance_matrix, checks=True)
            except Exception as e:
                logger.warning(f"Error in squareform: {e}, attempting without checks")
                condensed_distances = squareform(distance_matrix, checks=False)
            
            # Ensure no negative distances
            condensed_distances = np.maximum(condensed_distances, 0)
            
            # Perform hierarchical clustering
            # Use appropriate method based on data
            if self.method == 'ward' and np.any(distance_matrix < 0):
                logger.warning("Ward method requires non-negative distances, switching to average")
                method = 'average'
            else:
                method = self.method
            
            clusters = linkage(condensed_distances, method=method)
            
            # Validate clustering result
            if clusters.shape[0] != n - 1:
                raise ValueError("Invalid clustering result")
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            # Return simple sequential clustering as fallback
            n = distance_matrix.shape[0]
            clusters = np.zeros((n-1, 4))
            for i in range(n-1):
                clusters[i] = [i, i+1, 1, 2]
            return clusters
    
    def get_quasi_diag_order(self, clusters: np.ndarray, n_assets: int) -> List[int]:
        """
        Get quasi-diagonalization order from clusters.
        
        Args:
            clusters: Linkage matrix from hierarchical clustering
            n_assets: Number of assets
            
        Returns:
            List of asset indices in quasi-diagonal order
        """
        try:
            # Handle empty clusters
            if clusters.size == 0:
                return [0]  # Single asset
            
            # Initialize order tracking
            order = []
            processed = set()
            
            def _get_cluster_order(cluster_id):
                """Recursively get order from cluster tree"""
                # Prevent infinite recursion
                if len(order) >= n_assets:
                    return []
                    
                if cluster_id < n_assets:
                    # Leaf node (original asset)
                    if cluster_id not in processed:
                        processed.add(cluster_id)
                        return [cluster_id]
                    else:
                        return []
                else:
                    # Internal node
                    cluster_idx = int(cluster_id - n_assets)
                    
                    # Validate cluster index
                    if cluster_idx < 0 or cluster_idx >= len(clusters):
                        logger.warning(f"Invalid cluster index: {cluster_idx}")
                        return []
                    
                    left_child = int(clusters[cluster_idx, 0])
                    right_child = int(clusters[cluster_idx, 1])
                    
                    # Recursively get order from children
                    left_order = _get_cluster_order(left_child)
                    right_order = _get_cluster_order(right_child)
                    
                    return left_order + right_order
            
            # Start from root (last cluster)
            if len(clusters) > 0:
                root_id = len(clusters) - 1 + n_assets
                order = _get_cluster_order(root_id)
            
            # Ensure all assets are included
            if len(order) != n_assets:
                logger.warning(f"Incomplete order: {len(order)} != {n_assets}")
                # Add missing assets
                for i in range(n_assets):
                    if i not in order:
                        order.append(i)
            
            return order
            
        except Exception as e:
            logger.error(f"Error getting quasi-diagonal order: {e}")
            # Return simple sequential order as fallback
            return list(range(n_assets))
    
    def validate_indices(self, indices: List[int], n_assets: int) -> bool:
        """
        Validate that indices are a valid permutation.
        
        Args:
            indices: List of asset indices
            n_assets: Expected number of assets
            
        Returns:
            True if indices are valid
        """
        try:
            if len(indices) != n_assets:
                return False
            
            if set(indices) != set(range(n_assets)):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_cluster_labels(self, clusters: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Get cluster labels for assets.
        
        Args:
            clusters: Linkage matrix
            n_clusters: Desired number of clusters
            
        Returns:
            Array of cluster labels
        """
        try:
            if clusters.size == 0:
                # Single asset
                return np.array([1])
            
            # Get cluster labels using fcluster
            cluster_labels = fcluster(clusters, n_clusters, criterion='maxclust')
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error getting cluster labels: {e}")
            # Return all assets in one cluster
            n_assets = len(clusters) + 1
            return np.ones(n_assets, dtype=int)
    
    def get_cluster_allocation(self, asset_names: List[str], clusters: np.ndarray, 
                             n_clusters: int = 3) -> dict:
        """
        Get asset allocation by clusters.
        
        Args:
            asset_names: List of asset names
            clusters: Linkage matrix
            n_clusters: Desired number of clusters
            
        Returns:
            Dictionary mapping cluster IDs to asset lists
        """
        try:
            # Get cluster labels
            cluster_labels = self.get_cluster_labels(clusters, n_clusters)
            
            # Group assets by cluster
            cluster_allocation = {}
            for i, asset in enumerate(asset_names):
                cluster_id = int(cluster_labels[i])
                if cluster_id not in cluster_allocation:
                    cluster_allocation[cluster_id] = []
                cluster_allocation[cluster_id].append(asset)
            
            return cluster_allocation
            
        except Exception as e:
            logger.error(f"Error getting cluster allocation: {e}")
            # Return all assets in one cluster as fallback
            return {1: list(asset_names)}
    
    def calculate_cluster_distance(self, cluster1_indices: List[int], 
                                 cluster2_indices: List[int],
                                 distance_matrix: np.ndarray) -> float:
        """
        Calculate distance between two clusters.
        
        Args:
            cluster1_indices: Indices of assets in cluster 1
            cluster2_indices: Indices of assets in cluster 2
            distance_matrix: Full distance matrix
            
        Returns:
            Distance between clusters based on linkage method
        """
        try:
            distances = []
            
            for i in cluster1_indices:
                for j in cluster2_indices:
                    distances.append(distance_matrix[i, j])
            
            if not distances:
                return 0.0
            
            if self.method == 'single':
                return min(distances)
            elif self.method == 'complete':
                return max(distances)
            elif self.method == 'average':
                return np.mean(distances)
            else:
                # Default to average
                return np.mean(distances)
                
        except Exception as e:
            logger.error(f"Error calculating cluster distance: {e}")
            return 0.0
