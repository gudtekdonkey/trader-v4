"""
Telegram Notifications Package - Trading alert system
Provides Telegram bot integration for sending trade notifications, position
updates, error alerts, and daily summaries.

File: __init__.py
Modified: 2025-07-18
"""

from .tg_notifications import (
    TelegramNotifier,
    get_notifier,
    notify_trade,
    notify_position,
    notify_error,
    notify_daily_summary,
    notify_risk_alert
)

__all__ = [
    'TelegramNotifier',
    'get_notifier',
    'notify_trade',
    'notify_position',
    'notify_error',
    'notify_daily_summary',
    'notify_risk_alert'
]
