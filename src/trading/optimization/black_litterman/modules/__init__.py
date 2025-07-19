"""
Black-Litterman Optimization Modules

This package contains the modular components for Black-Litterman portfolio optimization.

Modules:
- matrix_operations: Covariance calculations, shrinkage, numerical stability
- bayesian_updater: Bayesian view incorporation
- portfolio_optimizer: Weight optimization with constraints
- view_generator: View generation from ML, technical analysis, sentiment
"""

from .matrix_operations import MatrixOperations
from .bayesian_updater import BayesianUpdater
from .portfolio_optimizer import PortfolioOptimizer
from .view_generator import ViewGenerator

__all__ = [
    'MatrixOperations',
    'BayesianUpdater',
    'PortfolioOptimizer',
    'ViewGenerator'
]
