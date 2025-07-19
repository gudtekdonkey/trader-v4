"""
Portfolio Monitoring and Alerting System - backward compatibility wrapper
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
