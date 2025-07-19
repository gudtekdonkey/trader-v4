"""
Data validation utilities
"""

import re
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates market data"""
    
    def __init__(self):
        self.valid_symbols = set()
        self.symbol_pattern = r'^[A-Z0-9]+-[A-Z0-9]+$'
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Check cache
        if symbol in self.valid_symbols:
            return True
        
        # Basic format validation
        if re.match(self.symbol_pattern, symbol):
            self.valid_symbols.add(symbol)
            return True
        
        return False
    
    def validate_price_size(self, price: Any, size: Any) -> bool:
        """Validate price and size values"""
        try:
            p = float(price)
            s = float(size)
            
            # Check for valid positive numbers
            if p <= 0 or s <= 0:
                return False
            
            # Check for reasonable ranges
            if p > 1e9 or s > 1e9:  # Sanity check
                return False
            
            # Check for NaN or Inf
            if not (np.isfinite(p) and np.isfinite(s)):
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def validate_orderbook_data(self, bids: list, asks: list) -> bool:
        """Validate orderbook data structure"""
        try:
            if not bids or not asks:
                return False
            
            # Check first level
            if len(bids[0]) != 2 or len(asks[0]) != 2:
                return False
            
            # Validate price levels are sorted
            for i in range(1, min(len(bids), 10)):
                if float(bids[i][0]) >= float(bids[i-1][0]):
                    return False
            
            for i in range(1, min(len(asks), 10)):
                if float(asks[i][0]) <= float(asks[i-1][0]):
                    return False
            
            return True
            
        except Exception:
            return False
