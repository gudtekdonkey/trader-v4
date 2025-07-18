"""
Order Validator Module

Handles comprehensive order validation including parameter checks,
risk limits, and exchange-specific requirements.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class OrderValidator:
    """
    Validates orders before submission to ensure compliance with
    trading rules and risk parameters.
    """
    
    def __init__(self):
        """Initialize the order validator."""
        self.validation_rules = self._initialize_validation_rules()
        self.symbol_specs = self._load_symbol_specifications()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules."""
        return {
            'min_notional': 10,  # Minimum $10 notional value
            'max_notional': 1000000,  # Maximum $1M notional value
            'max_price_deviation': 0.1,  # 10% max deviation from market
            'min_price_increment': 0.00001,  # Minimum price increment
            'min_size_increment': 0.00001,  # Minimum size increment
            'max_orders_per_symbol': 10,  # Max open orders per symbol
            'max_position_size': 100000,  # Max position size in base currency
        }
    
    def _load_symbol_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Load symbol-specific trading specifications."""
        # In practice, this would load from exchange API or config
        return {
            'BTC-PERP': {
                'min_size': 0.001,
                'max_size': 100,
                'tick_size': 0.1,
                'min_notional': 10
            },
            'ETH-PERP': {
                'min_size': 0.01,
                'max_size': 1000,
                'tick_size': 0.01,
                'min_notional': 10
            },
            # Default for unknown symbols
            'DEFAULT': {
                'min_size': 0.00001,
                'max_size': 1000000,
                'tick_size': 0.00001,
                'min_notional': 1
            }
        }
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float],
        params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate order parameters with comprehensive checks.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            size: Order size
            order_type: Order type
            price: Limit price
            params: Additional parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic parameter validation
            basic_validation = self._validate_basic_params(
                symbol, side, size, order_type, price
            )
            if not basic_validation[0]:
                return basic_validation
            
            # Symbol-specific validation
            symbol_validation = self._validate_symbol_specs(symbol, size, price)
            if not symbol_validation[0]:
                return symbol_validation
            
            # Size bounds validation
            size_validation = self._validate_size_bounds(size, params)
            if not size_validation[0]:
                return size_validation
            
            # Price validation for limit orders
            if order_type == 'limit' and price is not None:
                price_validation = self._validate_price(symbol, price)
                if not price_validation[0]:
                    return price_validation
            
            # Notional value validation
            if price is not None:
                notional_validation = self._validate_notional(size, price)
                if not notional_validation[0]:
                    return notional_validation
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False, "Validation error"
    
    def _validate_basic_params(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """Validate basic order parameters."""
        # Symbol validation
        if not symbol or not isinstance(symbol, str) or len(symbol) < 3:
            return False, "Invalid symbol"
        
        # Size validation
        if not isinstance(size, (int, float)) or size <= 0 or not np.isfinite(size):
            return False, "Invalid order size"
        
        # Order type validation
        if order_type not in ['market', 'limit']:
            return False, f"Invalid order type: {order_type}"
        
        # Price validation for limit orders
        if order_type == 'limit':
            if price is None or not isinstance(price, (int, float)) or price <= 0 or not np.isfinite(price):
                return False, "Limit order requires valid price"
        
        # Side validation
        if side not in ['buy', 'sell']:
            return False, f"Invalid order side: {side}"
        
        return True, None
    
    def _validate_symbol_specs(
        self,
        symbol: str,
        size: float,
        price: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """Validate order against symbol specifications."""
        specs = self.symbol_specs.get(symbol, self.symbol_specs['DEFAULT'])
        
        # Minimum size check
        if size < specs['min_size']:
            return False, f"Size below minimum: {specs['min_size']}"
        
        # Maximum size check
        if size > specs['max_size']:
            return False, f"Size above maximum: {specs['max_size']}"
        
        # Size increment check
        size_increment = specs.get('size_increment', 0.00001)
        if not self._is_valid_increment(size, size_increment):
            return False, f"Invalid size increment, must be multiple of {size_increment}"
        
        # Price tick size check
        if price is not None:
            tick_size = specs['tick_size']
            if not self._is_valid_increment(price, tick_size):
                return False, f"Invalid price tick, must be multiple of {tick_size}"
        
        return True, None
    
    def _validate_size_bounds(
        self,
        size: float,
        params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate size against configured bounds."""
        min_size = params.get('min_order_size', 1)
        max_size = params.get('max_order_size', 1000000)
        
        if size < min_size:
            return False, f"Size below minimum: {min_size}"
        
        if size > max_size:
            return False, f"Size exceeds maximum: {max_size}"
        
        return True, None
    
    def _validate_price(
        self,
        symbol: str,
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """Validate price for reasonableness."""
        # Price must be positive
        if price <= 0:
            return False, "Price must be positive"
        
        # Check for extreme prices (potential fat finger)
        if price > 1000000:  # $1M per unit
            return False, "Price unreasonably high"
        
        if price < 0.00001:  # $0.00001 per unit
            return False, "Price unreasonably low"
        
        return True, None
    
    def _validate_notional(
        self,
        size: float,
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """Validate notional value of the order."""
        notional = size * price
        
        if notional < self.validation_rules['min_notional']:
            return False, f"Notional value below minimum: ${self.validation_rules['min_notional']}"
        
        if notional > self.validation_rules['max_notional']:
            return False, f"Notional value exceeds maximum: ${self.validation_rules['max_notional']}"
        
        return True, None
    
    def _is_valid_increment(self, value: float, increment: float) -> bool:
        """Check if value is a valid multiple of increment."""
        if increment == 0:
            return True
        
        # Use modulo with tolerance for floating point precision
        remainder = value % increment
        tolerance = increment * 0.0001  # 0.01% tolerance
        
        return remainder < tolerance or (increment - remainder) < tolerance
    
    def validate_order_modification(
        self,
        original_order: Dict[str, Any],
        new_size: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate order modification request.
        
        Args:
            original_order: Original order details
            new_size: New size (if modifying)
            new_price: New price (if modifying)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Can't modify filled or cancelled orders
            if original_order['status'] in ['filled', 'cancelled', 'rejected']:
                return False, f"Cannot modify {original_order['status']} order"
            
            # Validate new size if provided
            if new_size is not None:
                if new_size <= 0:
                    return False, "New size must be positive"
                
                # Size can only be reduced
                if new_size > original_order['size']:
                    return False, "Cannot increase order size"
                
                # Must be greater than filled size
                if new_size <= original_order.get('filled_size', 0):
                    return False, "New size must be greater than filled size"
            
            # Validate new price if provided
            if new_price is not None and new_price <= 0:
                return False, "New price must be positive"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating order modification: {e}")
            return False, "Validation error"
    
    def validate_risk_limits(
        self,
        symbol: str,
        side: str,
        size: float,
        current_position: float,
        risk_limits: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate order against risk limits.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            size: Order size
            current_position: Current position size
            risk_limits: Risk limit parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Calculate resulting position
            if side == 'buy':
                resulting_position = current_position + size
            else:
                resulting_position = current_position - size
            
            # Check position limits
            max_position = risk_limits.get('max_position_size', 100000)
            if abs(resulting_position) > max_position:
                return False, f"Resulting position exceeds limit: {max_position}"
            
            # Check if order would flip position (risky)
            if current_position > 0 and resulting_position < 0:
                if abs(resulting_position) > max_position * 0.5:
                    return False, "Position flip too large"
            elif current_position < 0 and resulting_position > 0:
                if abs(resulting_position) > max_position * 0.5:
                    return False, "Position flip too large"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating risk limits: {e}")
            return False, "Risk validation error"
