"""
Black-Litterman Package - Bayesian portfolio optimization
Implements Black-Litterman model combining market equilibrium with investor
views using Bayesian inference for optimal portfolio allocation.

File: __init__.py
Modified: 2025-07-19
"""

from .black_litterman import BlackLittermanOptimizer

__all__ = ['BlackLittermanOptimizer']

__version__ = '1.0.0'
__author__ = 'Crypto Trading Bot Team'
