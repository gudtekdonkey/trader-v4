"""Telegram notification service - Main coordinator"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .modules.telegram_sender import TelegramSender, NotificationConfig
from .modules.message_formatter import MessageFormatter
from .modules.trading_notifications import TradingNotifications
from .modules.performance_notifications import PerformanceNotifications
from .modules.risk_notifications import RiskNotifications

from ..logger import setup_logger

logger = setup_logger(__name__)


class TelegramNotifier:
    """Telegram notification service for trading bot - Main coordinator"""
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig.from_env()
        self.sender = TelegramSender(self.config)
        self.formatter = MessageFormatter()
        
        # Initialize notification handlers
        self.trading = TradingNotifications(self.formatter)
        self.performance = PerformanceNotifications(self.formatter)
        self.risk = RiskNotifications(self.formatter)
        
        logger.info("TelegramNotifier initialized with all modules")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.sender.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.sender.__aexit__(exc_type, exc_val, exc_tb)
    
    # Trading notifications
    async def notify_trade_execution(self, trade_data: Dict[str, Any]) -> bool:
        """Notify about successful trade execution"""
        return await self.trading.notify_trade_execution(self.sender.send_message, trade_data)
    
    async def notify_position_update(self, position_data: Dict[str, Any], 
                                   action: str = "opened") -> bool:
        """Notify about position updates (open/close)"""
        return await self.trading.notify_position_update(self.sender.send_message, position_data, action)
    
    async def notify_strategy_signal(self, strategy: str, signal: Dict[str, Any]) -> bool:
        """Notify about strategy signals"""
        return await self.trading.notify_strategy_signal(self.sender.send_message, strategy, signal)
    
    # Performance notifications
    async def notify_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily performance summary"""
        return await self.performance.notify_daily_summary(self.sender.send_message, summary_data)
    
    # Risk notifications
    async def notify_error(self, error_type: str, error_message: str, 
                         context: Optional[Dict[str, Any]] = None) -> bool:
        """Notify about errors and exceptions"""
        async def sender_with_priority(message, disable_notification=False):
            return await self.sender.send_message(message, disable_notification=disable_notification)
        
        return await self.risk.notify_error(sender_with_priority, error_type, error_message, context)
    
    async def notify_risk_alert(self, alert_type: str, details: Dict[str, Any]) -> bool:
        """Send risk management alerts"""
        async def sender_with_priority(message, disable_notification=False):
            return await self.sender.send_message(message, disable_notification=disable_notification)
        
        return await self.risk.notify_risk_alert(sender_with_priority, alert_type, details)
    
    # Utility methods
    async def send_custom_message(self, message: str, parse_mode: str = 'HTML',
                                disable_notification: bool = None) -> bool:
        """Send custom message"""
        if disable_notification is None:
            disable_notification = self.config.silent_mode
        
        return await self.sender.send_message(message, parse_mode, disable_notification)
    
    async def test_connection(self) -> bool:
        """Test Telegram connection and credentials"""
        return await self.sender.test_connection()


# Singleton instance for easy access
_notifier_instance: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create singleton notifier instance"""
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = TelegramNotifier()
    return _notifier_instance


# Convenience functions for common notifications
async def notify_trade(trade_data: Dict[str, Any]) -> bool:
    """Convenience function to notify trade execution"""
    notifier = get_notifier()
    return await notifier.notify_trade_execution(trade_data)


async def notify_position(position_data: Dict[str, Any], action: str = "opened") -> bool:
    """Convenience function to notify position update"""
    notifier = get_notifier()
    return await notifier.notify_position_update(position_data, action)


async def notify_error(error_type: str, error_message: str, 
                      context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to notify errors"""
    notifier = get_notifier()
    return await notifier.notify_error(error_type, error_message, context)


async def notify_daily_summary(summary_data: Dict[str, Any]) -> bool:
    """Convenience function to send daily summary"""
    notifier = get_notifier()
    return await notifier.notify_daily_summary(summary_data)


async def notify_risk_alert(alert_type: str, details: Dict[str, Any]) -> bool:
    """Convenience function to send risk alerts"""
    notifier = get_notifier()
    return await notifier.notify_risk_alert(alert_type, details)


# Example usage and testing
if __name__ == "__main__":
    async def test_notifications():
        """Test all notification types"""
        notifier = TelegramNotifier()
        
        # Test connection
        print("Testing connection...")
        await notifier.test_connection()
        
        # Test trade notification
        print("Testing trade notification...")
        await notifier.notify_trade_execution({
            'symbol': 'BTC-USD',
            'side': 'buy',
            'price': 45000,
            'quantity': 0.1,
            'strategy': 'Momentum'
        })
        
        # Test position notification
        print("Testing position notification...")
        await notifier.notify_position_update({
            'symbol': 'ETH-USD',
            'side': 'long',
            'entry_price': 3000,
            'quantity': 2,
            'leverage': 5
        }, action="opened")
        
        # Test error notification
        print("Testing error notification...")
        await notifier.notify_error(
            "ConnectionError",
            "Failed to connect to exchange API",
            {'exchange': 'Hyperliquid', 'retry_count': 3}
        )
        
        # Test daily summary
        print("Testing daily summary...")
        await notifier.notify_daily_summary({
            'date': '2025-07-15',
            'total_trades': 25,
            'winning_trades': 18,
            'losing_trades': 7,
            'total_pnl': 2500.50,
            'win_rate': 72,
            'avg_win': 200,
            'avg_loss': 85,
            'largest_win': 800,
            'largest_loss': 250,
            'sharpe_ratio': 3.2,
            'max_drawdown': 8.5,
            'profit_factor': 2.4,
            'active_positions': [
                {'symbol': 'BTC-USD', 'side': 'long', 'unrealized_pnl': 450},
                {'symbol': 'SOL-USD', 'side': 'short', 'unrealized_pnl': -120}
            ]
        })
        
        print("All tests completed!")
    
    # Run tests
    import asyncio
    asyncio.run(test_notifications())
