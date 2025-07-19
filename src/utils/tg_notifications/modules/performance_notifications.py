"""Performance and summary notification handlers"""

from typing import Dict, Any
from datetime import datetime, timezone
from ...logger import setup_logger

logger = setup_logger(__name__)


class PerformanceNotifications:
    """Handles performance and summary notifications"""
    
    def __init__(self, formatter):
        self.formatter = formatter
    
    async def notify_daily_summary(self, sender, summary_data: Dict[str, Any]) -> bool:
        """Send daily performance summary"""
        try:
            date = summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            total_trades = summary_data.get('total_trades', 0)
            winning_trades = summary_data.get('winning_trades', 0)
            losing_trades = summary_data.get('losing_trades', 0)
            
            total_pnl = summary_data.get('total_pnl', 0)
            pnl_formatted = self.formatter.format_number(abs(total_pnl), 2, '$')
            pnl_sign = "+" if total_pnl >= 0 else "-"
            
            win_rate = summary_data.get('win_rate', 0)
            avg_win = self.formatter.format_number(summary_data.get('avg_win', 0), 2, '$')
            avg_loss = self.formatter.format_number(summary_data.get('avg_loss', 0), 2, '$')
            
            largest_win = self.formatter.format_number(summary_data.get('largest_win', 0), 2, '$')
            largest_loss = self.formatter.format_number(summary_data.get('largest_loss', 0), 2, '$')
            
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
• Win Rate: {self.formatter.format_percentage(win_rate)}

💰 <b>Profit & Loss</b>
• Total P&L: {pnl_sign}{pnl_formatted}
• Average Win: {avg_win}
• Average Loss: {avg_loss}
• Largest Win: {largest_win}
• Largest Loss: {largest_loss}

📈 <b>Portfolio Metrics</b>
• Sharpe Ratio: {self.formatter.format_number(summary_data.get('sharpe_ratio', 0), 2)}
• Max Drawdown: {self.formatter.format_percentage(summary_data.get('max_drawdown', 0))}
• Profit Factor: {self.formatter.format_number(summary_data.get('profit_factor', 0), 2)}
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
                    pnl_formatted = self.formatter.format_number(abs(unrealized_pnl), 2, '$')
                    
                    message += f"• {symbol} ({side}): {pnl_sign}{pnl_formatted}\n"
                
                if len(active_positions) > 5:
                    message += f"... and {len(active_positions) - 5} more positions\n"
            
            return await sender(message)
        except Exception as e:
            logger.error(f"Error formatting daily summary notification: {e}")
            return False
