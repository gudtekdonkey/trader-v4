"""Message formatting utilities"""

from typing import Any
from decimal import Decimal
import numpy as np


class MessageFormatter:
    """Handles message formatting for notifications"""
    
    def format_number(self, value: Any, decimals: int = 2, prefix: str = '') -> str:
        """Format number for display"""
        try:
            if isinstance(value, (int, float, Decimal)):
                formatted = f"{float(value):,.{decimals}f}"
                return f"{prefix}{formatted}" if prefix else formatted
            return str(value)
        except:
            return str(value)
    
    def format_percentage(self, value: Any) -> str:
        """Format percentage for display"""
        try:
            return f"{float(value):+.2f}%"
        except:
            return str(value)
