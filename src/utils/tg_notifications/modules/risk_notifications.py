"""Risk and error notification handlers"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from ...logger import setup_logger

logger = setup_logger(__name__)


class RiskNotifications:
    """Handles risk and error notifications"""
    
    def __init__(self, formatter):
        self.formatter = formatter
    
    async def notify_error(self, sender, error_type: str, error_message: str, 
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
            
            return await sender(message, disable_notification=False)
        except Exception as e:
            logger.error(f"Error formatting error notification: {e}")
            return False
    
    async def notify_risk_alert(self, sender, alert_type: str, details: Dict[str, Any]) -> bool:
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
                current_dd = self.formatter.format_percentage(details.get('current_drawdown', 0))
                limit = self.formatter.format_percentage(details.get('limit', 20))
                message += f"""
<b>Current Drawdown:</b> {current_dd}
<b>Limit:</b> {limit}
<b>Action:</b> Trading halted until drawdown recovers
"""
            
            elif alert_type == 'position_size':
                symbol = details.get('symbol', 'N/A')
                proposed_size = self.formatter.format_percentage(details.get('proposed_size', 0))
                max_size = self.formatter.format_percentage(details.get('max_size', 2))
                message += f"""
<b>Symbol:</b> {symbol}
<b>Proposed Size:</b> {proposed_size} of capital
<b>Max Allowed:</b> {max_size}
<b>Action:</b> Position size reduced to comply with limits
"""
            
            elif alert_type == 'liquidation':
                symbol = details.get('symbol', 'N/A')
                current_price = self.formatter.format_number(details.get('current_price', 0), 4, '$')
                liq_price = self.formatter.format_number(details.get('liquidation_price', 0), 4, '$')
                distance = self.formatter.format_percentage(details.get('distance_percentage', 0))
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
            
            return await sender(message, disable_notification=False)
        except Exception as e:
            logger.error(f"Error formatting risk alert notification: {e}")
            return False
