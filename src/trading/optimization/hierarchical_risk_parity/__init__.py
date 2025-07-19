"""
Hierarchical Risk Parity Package - Clustering-based portfolio optimization
Implements HRP algorithm using hierarchical clustering to build diversified
portfolios with equal risk allocation across correlated asset clusters.

File: __init__.py
Modified: 2025-07-19
"""

from .hierarchical_risk_parity import HRPOptimizer

__all__ = ['HRPOptimizer']

__version__ = '1.0.0'
__author__ = 'Crypto Trading Bot Team'
