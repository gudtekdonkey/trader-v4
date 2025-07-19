"""
Portfolio Optimization Package

This package contains advanced portfolio optimization algorithms for the trading bot.

Modules:
- black_litterman: Black-Litterman Bayesian portfolio optimization
- hierarchical_risk_parity: HRP clustering-based portfolio optimization
"""

from .black_litterman import BlackLittermanOptimizer
from .hierarchical_risk_parity import HRPOptimizer

__all__ = ['BlackLittermanOptimizer', 'HRPOptimizer']
