"""
Order Types Module

Defines order data structures and enumerations used throughout
the order execution system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any
import time


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date
    DAY = "DAY"  # Day order


@dataclass
class Order:
    """
    Order data structure with validation.
    
    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        side: Trade side (buy/sell)
        size: Order size
        order_type: Order type (market/limit)
        price: Limit price (optional)
        status: Current order status
        filled_size: Amount filled
        avg_fill_price: Average execution price
        timestamp: Order creation timestamp
        time_in_force: Time in force instruction
        post_only: Post-only flag for maker orders
        reduce_only: Reduce-only flag
        metadata: Additional order metadata
    """
    order_id: str
    symbol: str
    side: str
    size: float
    order_type: str
    price: Optional[float]
    status: OrderStatus
    filled_size: float
    avg_fill_price: float
    timestamp: float
    time_in_force: str
    post_only: bool
    reduce_only: bool
    metadata: Dict[str, Any]
    
    # Additional tracking fields
    last_update_time: float = 0
    exchange_order_id: Optional[str] = None
    fees_paid: float = 0
    slippage: float = 0
    rejection_reason: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.last_update_time == 0:
            self.last_update_time = self.timestamp
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.PENDING, 
            OrderStatus.SUBMITTED, 
            OrderStatus.PARTIAL
        ]
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in [
            OrderStatus.FILLED, 
            OrderStatus.CANCELLED, 
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def remaining_size(self) -> float:
        """Calculate remaining size to fill."""
        return max(0, self.size - self.filled_size)
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.size == 0:
            return 0
        return (self.filled_size / self.size) * 100
    
    @property
    def age_seconds(self) -> float:
        """Calculate order age in seconds."""
        return time.time() - self.timestamp
    
    def update_fill(self, filled_size: float, avg_price: float) -> None:
        """
        Update order fill information.
        
        Args:
            filled_size: Total filled size
            avg_price: Average fill price
        """
        self.filled_size = filled_size
        self.avg_fill_price = avg_price
        self.last_update_time = time.time()
        
        if self.filled_size >= self.size:
            self.status = OrderStatus.FILLED
        elif self.filled_size > 0:
            self.status = OrderStatus.PARTIAL


@dataclass
class ExecutionReport:
    """
    Execution report for completed orders.
    
    Attributes:
        order_id: Order identifier
        symbol: Trading symbol
        side: Trade side
        executed_size: Size executed
        executed_price: Average execution price
        fees: Total fees paid
        slippage: Slippage from expected price
        execution_time: Time to execute
        venue: Execution venue
    """
    order_id: str
    symbol: str
    side: str
    executed_size: float
    executed_price: float
    fees: float
    slippage: float
    execution_time: float
    venue: str
    timestamp: float = 0
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class OrderRequest:
    """
    Order request parameters.
    
    Used to validate and prepare orders before submission.
    """
    symbol: str
    side: str
    size: float
    order_type: str = 'limit'
    price: Optional[float] = None
    time_in_force: str = 'GTC'
    post_only: bool = False
    reduce_only: bool = False
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate order request parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Symbol validation
        if not self.symbol or not isinstance(self.symbol, str):
            return False, "Invalid symbol"
        
        # Side validation
        if self.side not in ['buy', 'sell']:
            return False, f"Invalid side: {self.side}"
        
        # Size validation
        if not isinstance(self.size, (int, float)) or self.size <= 0:
            return False, "Invalid size"
        
        # Order type validation
        if self.order_type not in ['market', 'limit', 'stop', 'stop_limit']:
            return False, f"Invalid order type: {self.order_type}"
        
        # Price validation for limit orders
        if self.order_type in ['limit', 'stop_limit']:
            if self.price is None or self.price <= 0:
                return False, "Limit order requires valid price"
        
        # Stop price validation
        if self.order_type in ['stop', 'stop_limit']:
            if self.stop_price is None or self.stop_price <= 0:
                return False, "Stop order requires valid stop price"
        
        # Time in force validation
        valid_tif = ['GTC', 'IOC', 'FOK', 'GTD', 'DAY']
        if self.time_in_force not in valid_tif:
            return False, f"Invalid time in force: {self.time_in_force}"
        
        return True, None
