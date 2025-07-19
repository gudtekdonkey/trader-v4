"""
Mean Reversion Strategy Package - Statistical arbitrage trading
Implements mean reversion strategies that profit from price deviations
returning to their statistical mean values.

File: __init__.py
Modified: 2025-07-19
"""

from .mean_reversion import MeanReversionStrategy, MeanReversionSignal

__all__ = ['MeanReversionStrategy', 'MeanReversionSignal']
