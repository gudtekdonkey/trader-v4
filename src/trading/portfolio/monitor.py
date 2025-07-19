"""
Portfolio Monitor Wrapper - Backward compatibility module
Provides compatibility layer for refactored portfolio monitoring module to
maintain existing import paths.

File: monitor.py
Modified: 2025-07-19
"""

from .monitor.monitor import (
    PortfolioMonitor,
    LogAlertHandler,
    EmailAlertHandler,
    SlackAlertHandler
)
from .monitor.modules.alert_manager import Alert, AlertSeverity, AlertType

# Export at module level for backward compatibility
__all__ = [
    'PortfolioMonitor',
    'Alert',
    'AlertSeverity',
    'AlertType',
    'LogAlertHandler',
    'EmailAlertHandler',
    'SlackAlertHandler'
]
