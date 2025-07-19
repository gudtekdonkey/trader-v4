"""Trading notification formatters and handlers"""

from typing import Dict, Any
from datetime import datetime, timezone
from decimal import Decimal
from ...logger import setup_logger

logger = setup_logger(__name__)


class TradingNotifications:
    """Handles trading-related notifications"""
    
    def __init__(self, formatter):
        self.formatter = formatter
    
    async def notify_trade_execution(self, sender, trade_data: Dict[str, Any]) -> bool:
        """Notify about successful trade execution"""
        try:
            symbol = trade_data.get('symbol', 'N/A')
            side = trade_data.get('side', 'N/A').upper()
            price = self.formatter.format_number(trade_data.get('price', 0), 4, '$')
            quantity = self.formatter.format_number(trade_data.get('quantity', 0), 4)
            value = self.formatter.format_number(
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
            
            return await sender(message)
        except Exception as e:
            logger.error(f"Error formatting trade execution notification: {e}")
            return False
    
    async def notify_position_update(self, sender, position_data: Dict[str, Any], 
                                   action: str = "opened") -> bool:
        """Notify about position updates (open/close)"""
        try:
            symbol = position_data.get('symbol', 'N/A')
            side = position_data.get('side', 'N/A').upper()
            entry_price = self.formatter.format_number(position_data.get('entry_price', 0), 4, '$')
            quantity = self.formatter.format_number(position_data.get('quantity', 0), 4)
            
            if action == "closed":
                exit_price = self.formatter.format_number(position_data.get('exit_price', 0), 4, '$')
                pnl = position_data.get('realized_pnl', 0)
                pnl_formatted = self.formatter.format_number(abs(pnl), 2, '$')
                pnl_percentage = self.formatter.format_percentage(
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
            
            return await sender(message)
        except Exception as e:
            logger.error(f"Error formatting position update notification: {e}")
            return False
    
    async def notify_strategy_signal(self, sender, strategy: str, signal: Dict[str, Any]) -> bool:
        """Notify about strategy signals"""
        try:
            symbol = signal.get('symbol', 'N/A')
            action = signal.get('action', 'N/A').upper()
            confidence = self.formatter.format_percentage(signal.get('confidence', 0))
            
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
                message += f"<b>Entry Price:</b> {self.formatter.format_number(signal['entry_price'], 4, '$')}\n"
            
            if 'stop_loss' in signal:
                message += f"<b>Stop Loss:</b> {self.formatter.format_number(signal['stop_loss'], 4, '$')}\n"
            
            if 'take_profit' in signal:
                message += f"<b>Take Profit:</b> {self.formatter.format_number(signal['take_profit'], 4, '$')}\n"
            
            if 'reason' in signal:
                message += f"<b>Reason:</b> {signal['reason']}\n"
            
            message += f"""
<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            return await sender(message)
        except Exception as e:
            logger.error(f"Error formatting strategy signal notification: {e}")
            return False
