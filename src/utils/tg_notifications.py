# utils/notifications.py

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass
import aiohttp
import json
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for Telegram notifications"""
    bot_token: str
    chat_id: str
    enabled: bool = True
    silent_mode: bool = False
    max_retries: int = 3
    retry_delay: int = 5
    
    @classmethod
    def from_env(cls) -> 'NotificationConfig':
        """Create config from environment variables"""
        return cls(
            bot_token=os.getenv('TELEGRAM_TOKEN', ''),
            chat_id=os.getenv('TELEGRAM_CHAT_ID', ''),
            enabled=os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true',
            silent_mode=os.getenv('TELEGRAM_SILENT', 'false').lower() == 'true'
        )


class TelegramNotifier:
    """Telegram notification service for trading bot"""
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig.from_env()
        self.base_url = f"https://api.telegram.org/bot{self.config.bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
        if not self.config.bot_token or not self.config.chat_id:
            logger.warning("Telegram credentials not configured. Notifications disabled.")
            self.config.enabled = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _send_message(self, text: str, parse_mode: str = 'HTML', 
                          disable_notification: bool = False) -> bool:
        """Send message to Telegram"""
        if not self.config.enabled:
            return False
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.config.chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_notification': disable_notification or self.config.silent_mode
        }
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(url, json=data) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {error_text}")
            except Exception as e:
                logger.error(f"Error sending Telegram message: {e}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay)
        
        return False
    
    def _format_number(self, value: Any, decimals: int = 2, prefix: str = '') -> str:
        """Format number for display"""
        try:
            if isinstance(value, (int, float, Decimal)):
                formatted = f"{float(value):,.{decimals}f}"
                return f"{prefix}{formatted}" if prefix else formatted
            return str(value)
        except:
            return str(value)
    
    def _format_percentage(self, value: Any) -> str:
        """Format percentage for display"""
        try:
            return f"{float(value):+.2f}%"
        except:
            return str(value)
    
    async def notify_trade_execution(self, trade_data: Dict[str, Any]) -> bool:
        """Notify about successful trade execution"""
        try:
            symbol = trade_data.get('symbol', 'N/A')
            side = trade_data.get('side', 'N/A').upper()
            price = self._format_number(trade_data.get('price', 0), 4, '$')
            quantity = self._format_number(trade_data.get('quantity', 0), 4)
            value = self._format_number(
                float(trade_data.get('price', 0)) * float(trade_data.get('quantity', 0)), 
                2, '$'
            )
            strategy = trade_data.get('strategy', 'Manual')
            
            # Determine emoji based on side
            emoji = "🟢" if side == "BUY" else "🔴"
            
            message = f"""
{emoji} <b>Trade Executed</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Price:</b> {price}
<b>Quantity:</b> {quantity}
<b>Value:</b> {value}
<b>Strategy:</b> {strategy}
<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            return await self._send_message(message)
        except Exception as e:
            logger.error(f"Error formatting trade execution notification: {e}")
            return False
    
    async def notify_position_update(self, position_data: Dict[str, Any], 
                                   action: str = "opened") -> bool:
        """Notify about position updates (open/close)"""
        try:
            symbol = position_data.get('symbol', 'N/A')
            side = position_data.get('side', 'N/A').upper()
            entry_price = self._format_number(position_data.get('entry_price', 0), 4, '$')
            quantity = self._format_number(position_data.get('quantity', 0), 4)
            
            if action == "closed":
                exit_price = self._format_number(position_data.get('exit_price', 0), 4, '$')
                pnl = position_data.get('realized_pnl', 0)
                pnl_formatted = self._format_number(abs(pnl), 2, '$')
                pnl_percentage = self._format_percentage(
                    position_data.get('pnl_percentage', 0)
                )
                
                # Determine emoji based on profit/loss
                emoji = "💰" if pnl >= 0 else "💸"
                pnl_sign = "+" if pnl >= 0 else "-"
                
                message = f"""
{emoji} <b>Position Closed</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry Price:</b> {entry_price}
<b>Exit Price:</b> {exit_price}
<b>Quantity:</b> {quantity}
<b>P&L:</b> {pnl_sign}{pnl_formatted} ({pnl_percentage})
<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            else:
                leverage = position_data.get('leverage', 1)
                message = f"""
📊 <b>Position Opened</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry Price:</b> {entry_price}
<b>Quantity:</b> {quantity}
<b>Leverage:</b> {leverage}x
<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            return await self._send_message(message)
        except Exception as e:
            logger.error(f"Error formatting position update notification: {e}")
            return False
    
    async def notify_error(self, error_type: str, error_message: str, 
                         context: Optional[Dict[str, Any]] = None) -> bool:
        """Notify about errors and exceptions"""
        try:
            context_str = ""
            if context:
                context_str = "\n<b>Context:</b>\n"
                for key, value in context.items():
                    context_str += f"  • {key}: {value}\n"
            
            message = f"""
⚠️ <b>Error Alert</b>

<b>Type:</b> {error_type}
<b>Message:</b> {error_message}
{context_str}
<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            return await self._send_message(message, disable_notification=False)
        except Exception as e:
            logger.error(f"Error formatting error notification: {e}")
            return False
    
    async def notify_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily performance summary"""
        try:
            date = summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            total_trades = summary_data.get('total_trades', 0)
            winning_trades = summary_data.get('winning_trades', 0)
            losing_trades = summary_data.get('losing_trades', 0)
            
            total_pnl = summary_data.get('total_pnl', 0)
            pnl_formatted = self._format_number(abs(total_pnl), 2, '$')
            pnl_sign = "+" if total_pnl >= 0 else "-"
            
            win_rate = summary_data.get('win_rate', 0)
            avg_win = self._format_number(summary_data.get('avg_win', 0), 2, '$')
            avg_loss = self._format_number(summary_data.get('avg_loss', 0), 2, '$')
            
            largest_win = self._format_number(summary_data.get('largest_win', 0), 2, '$')
            largest_loss = self._format_number(summary_data.get('largest_loss', 0), 2, '$')
            
            # Determine overall emoji
            if total_pnl > 0:
                emoji = "🎉"
            elif total_pnl < 0:
                emoji = "😔"
            else:
                emoji = "😐"
            
            message = f"""
{emoji} <b>Daily Performance Summary</b>

<b>Date:</b> {date}

📊 <b>Trading Activity</b>
• Total Trades: {total_trades}
• Winning Trades: {winning_trades}
• Losing Trades: {losing_trades}
• Win Rate: {self._format_percentage(win_rate)}

💰 <b>Profit & Loss</b>
• Total P&L: {pnl_sign}{pnl_formatted}
• Average Win: {avg_win}
• Average Loss: {avg_loss}
• Largest Win: {largest_win}
• Largest Loss: {largest_loss}

📈 <b>Portfolio Metrics</b>
• Sharpe Ratio: {self._format_number(summary_data.get('sharpe_ratio', 0), 2)}
• Max Drawdown: {self._format_percentage(summary_data.get('max_drawdown', 0))}
• Profit Factor: {self._format_number(summary_data.get('profit_factor', 0), 2)}
"""
            
            # Add active positions if any
            active_positions = summary_data.get('active_positions', [])
            if active_positions:
                message += "\n\n📍 <b>Active Positions</b>\n"
                for pos in active_positions[:5]:  # Limit to 5 positions
                    symbol = pos.get('symbol', 'N/A')
                    side = pos.get('side', 'N/A').upper()
                    unrealized_pnl = pos.get('unrealized_pnl', 0)
                    pnl_sign = "+" if unrealized_pnl >= 0 else "-"
                    pnl_formatted = self._format_number(abs(unrealized_pnl), 2, '$')
                    
                    message += f"• {symbol} ({side}): {pnl_sign}{pnl_formatted}\n"
                
                if len(active_positions) > 5:
                    message += f"... and {len(active_positions) - 5} more positions\n"
            
            return await self._send_message(message)
        except Exception as e:
            logger.error(f"Error formatting daily summary notification: {e}")
            return False
    
    async def notify_risk_alert(self, alert_type: str, details: Dict[str, Any]) -> bool:
        """Send risk management alerts"""
        try:
            alert_messages = {
                'max_drawdown': "Maximum drawdown limit reached!",
                'position_size': "Position size exceeds risk limits!",
                'correlation': "High correlation detected in portfolio!",
                'margin_call': "Margin call warning!",
                'liquidation': "Liquidation risk detected!"
            }
            
            alert_title = alert_messages.get(alert_type, "Risk Alert")
            
            message = f"""
🚨 <b>{alert_title}</b>

<b>Alert Type:</b> {alert_type}
"""
            
            # Add specific details based on alert type
            if alert_type == 'max_drawdown':
                current_dd = self._format_percentage(details.get('current_drawdown', 0))
                limit = self._format_percentage(details.get('limit', 20))
                message += f"""
<b>Current Drawdown:</b> {current_dd}
<b>Limit:</b> {limit}
<b>Action:</b> Trading halted until drawdown recovers
"""
            
            elif alert_type == 'position_size':
                symbol = details.get('symbol', 'N/A')
                proposed_size = self._format_percentage(details.get('proposed_size', 0))
                max_size = self._format_percentage(details.get('max_size', 2))
                message += f"""
<b>Symbol:</b> {symbol}
<b>Proposed Size:</b> {proposed_size} of capital
<b>Max Allowed:</b> {max_size}
<b>Action:</b> Position size reduced to comply with limits
"""
            
            elif alert_type == 'liquidation':
                symbol = details.get('symbol', 'N/A')
                current_price = self._format_number(details.get('current_price', 0), 4, '$')
                liq_price = self._format_number(details.get('liquidation_price', 0), 4, '$')
                distance = self._format_percentage(details.get('distance_percentage', 0))
                message += f"""
<b>Symbol:</b> {symbol}
<b>Current Price:</b> {current_price}
<b>Liquidation Price:</b> {liq_price}
<b>Distance:</b> {distance}
<b>Action:</b> Consider reducing position or adding margin
"""
            
            message += f"""

<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            return await self._send_message(message, disable_notification=False)
        except Exception as e:
            logger.error(f"Error formatting risk alert notification: {e}")
            return False
    
    async def notify_strategy_signal(self, strategy: str, signal: Dict[str, Any]) -> bool:
        """Notify about strategy signals"""
        try:
            symbol = signal.get('symbol', 'N/A')
            action = signal.get('action', 'N/A').upper()
            confidence = self._format_percentage(signal.get('confidence', 0))
            
            # Determine emoji based on action
            emoji_map = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '⏸️',
                'CLOSE': '🏁'
            }
            emoji = emoji_map.get(action, '📊')
            
            message = f"""
{emoji} <b>Strategy Signal</b>

<b>Strategy:</b> {strategy}
<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Confidence:</b> {confidence}
"""
            
            # Add signal details if available
            if 'entry_price' in signal:
                message += f"<b>Entry Price:</b> {self._format_number(signal['entry_price'], 4, '$')}\n"
            
            if 'stop_loss' in signal:
                message += f"<b>Stop Loss:</b> {self._format_number(signal['stop_loss'], 4, '$')}\n"
            
            if 'take_profit' in signal:
                message += f"<b>Take Profit:</b> {self._format_number(signal['take_profit'], 4, '$')}\n"
            
            if 'reason' in signal:
                message += f"<b>Reason:</b> {signal['reason']}\n"
            
            message += f"""
<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            return await self._send_message(message)
        except Exception as e:
            logger.error(f"Error formatting strategy signal notification: {e}")
            return False
    
    async def send_custom_message(self, message: str, parse_mode: str = 'HTML',
                                disable_notification: bool = None) -> bool:
        """Send custom message"""
        if disable_notification is None:
            disable_notification = self.config.silent_mode
        
        return await self._send_message(message, parse_mode, disable_notification)
    
    async def test_connection(self) -> bool:
        """Test Telegram connection and credentials"""
        test_message = """
✅ <b>Telegram Notification Test</b>

Your Telegram notifications are configured correctly!
Bot is ready to send trading alerts.

<b>Time:</b> {}
""".format(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))
        
        success = await self._send_message(test_message)
        if success:
            logger.info("Telegram connection test successful")
        else:
            logger.error("Telegram connection test failed")
        
        return success


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
    asyncio.run(test_notifications())