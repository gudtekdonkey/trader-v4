"""
Portfolio Analytics Wrapper - Backward compatibility module
Provides compatibility layer for refactored portfolio analytics module to
maintain existing import paths.

File: analytics.py
Modified: 2025-07-19
"""

from .analytics.analytics import PortfolioAnalytics
from .analytics.modules.data_types import PortfolioMetrics, RebalancingRecommendation

# Export at module level for backward compatibility
__all__ = ['PortfolioAnalytics', 'PortfolioMetrics', 'RebalancingRecommendation']
