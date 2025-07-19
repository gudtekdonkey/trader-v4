"""
Adaptive Strategy Manager Modules

This package contains the modular components for adaptive strategy allocation.

Modules:
- strategy_allocation: Regime-based allocation and risk calculations
- performance_tracker: Performance tracking and scoring
- allocation_adjuster: Dynamic adjustments and normalization
"""

from .strategy_allocation import StrategyAllocator
from .performance_tracker import PerformanceTracker
from .allocation_adjuster import AllocationAdjuster

__all__ = [
    'StrategyAllocator',
    'PerformanceTracker',
    'AllocationAdjuster'
]
