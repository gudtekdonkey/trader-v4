"""
Portfolio Analytics Module - backward compatibility wrapper
"""

from .analytics.analytics import PortfolioAnalytics
from .analytics.modules.data_types import PortfolioMetrics, RebalancingRecommendation

# Export at module level for backward compatibility
__all__ = ['PortfolioAnalytics', 'PortfolioMetrics', 'RebalancingRecommendation']
