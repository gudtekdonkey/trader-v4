"""
Advanced Executor Modules

This package contains the modular components for advanced order execution algorithms.

Modules:
- execution_algorithms: TWAP, VWAP, and Iceberg execution logic
- order_manager: Child order placement and tracking
- volume_profile: Volume data processing for VWAP
- execution_analytics: Performance tracking and statistics
"""

from .execution_algorithms import (
    TWAPExecutor,
    VWAPExecutor,
    IcebergExecutor,
    ExecutionAlgorithm
)
from .order_manager import OrderManager
from .volume_profile import VolumeProfileManager
from .execution_analytics import ExecutionAnalytics

__all__ = [
    'TWAPExecutor',
    'VWAPExecutor',
    'IcebergExecutor',
    'ExecutionAlgorithm',
    'OrderManager',
    'VolumeProfileManager',
    'ExecutionAnalytics'
]
