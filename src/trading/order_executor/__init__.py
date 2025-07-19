"""
Order Executor Package - Advanced order execution system
Provides sophisticated order execution algorithms including TWAP, VWAP,
iceberg orders, and smart routing for optimal trade execution.

File: __init__.py
Modified: 2025-07-18
"""

from .order_executor import OrderExecutor

__all__ = ['OrderExecutor']
