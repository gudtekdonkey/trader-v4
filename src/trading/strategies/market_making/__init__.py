"""
Market Making Strategy Package - Liquidity provision trading
Implements market making strategies that provide liquidity by placing
limit orders on both sides of the order book.

File: __init__.py
Modified: 2025-07-19
"""

from .market_making import MarketMakingStrategy

__all__ = ['MarketMakingStrategy']
