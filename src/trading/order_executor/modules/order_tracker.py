"""
Order Tracker Module

Manages order lifecycle, tracking, and state management for all orders.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

from .order_types import Order, OrderStatus, ExecutionReport

logger = logging.getLogger(__name__)


class OrderTracker:
    """
    Tracks and manages order lifecycle and state.
    
    Maintains active orders, order history, and provides
    order state management functionality.
    """
    
    def __init__(self):
        """Initialize the order tracker."""
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.execution_reports: List[ExecutionReport] = []
        
        # Order indexing for fast lookups
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_status: Dict[OrderStatus, List[str]] = defaultdict(list)
        
        # Statistics
        self.order_count = 0
        self.last_order_id = 0
    
    def create_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float],
        time_in_force: str,
        post_only: bool,
        reduce_only: bool,
        metadata: Optional[Dict]
    ) -> Order:
        """
        Create and register a new order.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            size: Order size
            order_type: Order type
            price: Limit price
            time_in_force: Time in force
            post_only: Post-only flag
            reduce_only: Reduce-only flag
            metadata: Additional metadata
            
        Returns:
            Created Order object
        """
        # Generate order ID
        order_id = self._generate_order_id()
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            order_type=order_type,
            price=price,
            status=OrderStatus.PENDING,
            filled_size=0,
            avg_fill_price=0,
            timestamp=time.time(),
            time_in_force=time_in_force,
            post_only=post_only,
            reduce_only=reduce_only,
            metadata=metadata or {}
        )
        
        # Register order
        self._register_order(order)
        
        logger.info(f"Created order {order_id}: {side} {size} {symbol}")
        
        return order
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.last_order_id += 1
        timestamp = int(time.time() * 1000)
        return f"ORD_{timestamp}_{self.last_order_id}"
    
    def _register_order(self, order: Order) -> None:
        """Register order in tracking systems."""
        self.active_orders[order.order_id] = order
        self.orders_by_symbol[order.symbol].append(order.order_id)
        self.orders_by_status[order.status].append(order.order_id)
        self.order_count += 1
    
    def update_order_status(
        self,
        order_id: str,
        new_status: OrderStatus,
        filled_size: Optional[float] = None,
        avg_price: Optional[float] = None
    ) -> bool:
        """
        Update order status.
        
        Args:
            order_id: Order ID to update
            new_status: New order status
            filled_size: Filled size (if applicable)
            avg_price: Average fill price (if applicable)
            
        Returns:
            True if update successful
        """
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            order = self.active_orders[order_id]
            old_status = order.status
            
            # Update status
            order.status = new_status
            order.last_update_time = time.time()
            
            # Update fill information if provided
            if filled_size is not None:
                order.filled_size = filled_size
            if avg_price is not None:
                order.avg_fill_price = avg_price
            
            # Update indexes
            self._update_status_index(order_id, old_status, new_status)
            
            logger.info(
                f"Updated order {order_id}: {old_status} -> {new_status}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False
    
    def _update_status_index(
        self,
        order_id: str,
        old_status: OrderStatus,
        new_status: OrderStatus
    ) -> None:
        """Update status-based order index."""
        if order_id in self.orders_by_status[old_status]:
            self.orders_by_status[old_status].remove(order_id)
        self.orders_by_status[new_status].append(order_id)
    
    def process_order_update(
        self,
        order: Order,
        status_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process order status update from exchange.
        
        Args:
            order: Order object
            status_update: Status update from exchange
            
        Returns:
            Processing result with action recommendations
        """
        try:
            exchange_status = status_update.get('status', '').lower()
            filled_size = status_update.get('filled_size', 0)
            avg_price = status_update.get('avg_fill_price', 0)
            
            # Map exchange status to internal status
            status_mapping = {
                'filled': OrderStatus.FILLED,
                'partial': OrderStatus.PARTIAL,
                'cancelled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
                'expired': OrderStatus.EXPIRED,
                'new': OrderStatus.SUBMITTED,
                'pending': OrderStatus.PENDING
            }
            
            new_status = status_mapping.get(exchange_status, order.status)
            
            # Update order
            order.status = new_status
            order.filled_size = filled_size
            order.avg_fill_price = avg_price
            order.last_update_time = time.time()
            
            # Determine if order is complete
            complete = new_status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED
            ]
            
            # Check if should replace with aggressive order
            should_replace = False
            if new_status == OrderStatus.PARTIAL:
                fill_rate = order.fill_percentage / 100
                age = order.age_seconds
                
                # Replace if fill rate is low and order is old
                if fill_rate < 0.5 and age > 10:
                    should_replace = True
            
            return {
                'complete': complete,
                'should_replace': should_replace,
                'fill_rate': order.fill_percentage / 100,
                'age': order.age_seconds
            }
            
        except Exception as e:
            logger.error(f"Error processing order update: {e}")
            return {
                'complete': False,
                'should_replace': False,
                'error': str(e)
            }
    
    def archive_order(self, order: Order) -> None:
        """
        Move order from active to history.
        
        Args:
            order: Order to archive
        """
        try:
            if order.order_id in self.active_orders:
                # Remove from active orders
                del self.active_orders[order.order_id]
                
                # Remove from symbol index
                if order.order_id in self.orders_by_symbol[order.symbol]:
                    self.orders_by_symbol[order.symbol].remove(order.order_id)
                
                # Remove from status index
                if order.order_id in self.orders_by_status[order.status]:
                    self.orders_by_status[order.status].remove(order.order_id)
                
                # Add to history
                self.order_history.append(order)
                
                # Create execution report if filled
                if order.status == OrderStatus.FILLED:
                    self._create_execution_report(order)
                
                # Limit history size
                if len(self.order_history) > 10000:
                    self.order_history = self.order_history[-5000:]
                
                logger.info(f"Archived order {order.order_id}")
                
        except Exception as e:
            logger.error(f"Error archiving order: {e}")
    
    def _create_execution_report(self, order: Order) -> None:
        """Create execution report for completed order."""
        report = ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            executed_size=order.filled_size,
            executed_price=order.avg_fill_price,
            fees=order.fees_paid,
            slippage=order.slippage,
            execution_time=order.last_update_time - order.timestamp,
            venue='hyperliquid'  # Could be dynamic
        )
        
        self.execution_reports.append(report)
        
        # Limit report history
        if len(self.execution_reports) > 1000:
            self.execution_reports = self.execution_reports[-500:]
    
    def get_active_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """
        Get active orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of active orders
        """
        if symbol:
            order_ids = self.orders_by_symbol.get(symbol, [])
            return [
                self.active_orders[oid] 
                for oid in order_ids 
                if oid in self.active_orders
            ]
        else:
            return list(self.active_orders.values())
    
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None
        """
        # Check active orders first
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Check history
        for order in reversed(self.order_history):
            if order.order_id == order_id:
                return order
        
        return None
    
    def handle_timeout(
        self,
        order: Order,
        client: Any
    ) -> Dict[str, Any]:
        """
        Handle order timeout.
        
        Args:
            order: Order that timed out
            client: Exchange client
            
        Returns:
            Timeout handling result
        """
        try:
            if order.filled_size > 0:
                order.status = OrderStatus.PARTIAL
            else:
                order.status = OrderStatus.EXPIRED
            
            # Try to cancel remaining
            if order.is_active:
                try:
                    asyncio.create_task(client.cancel_order(order.order_id))
                except Exception as e:
                    logger.warning(f"Failed to cancel expired order: {e}")
            
            return {
                'status': order.status.value,
                'filled_size': order.filled_size,
                'avg_fill_price': order.avg_fill_price,
                'timeout': True
            }
            
        except Exception as e:
            logger.error(f"Error handling timeout: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'filled_size': order.filled_size
            }
    
    def get_statistics(
        self,
        symbol: Optional[str] = None,
        time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get order tracking statistics.
        
        Args:
            symbol: Optional symbol filter
            time_window: Optional time window in seconds
            
        Returns:
            Statistics dictionary
        """
        try:
            # Filter orders
            orders = self.order_history
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            
            if time_window:
                cutoff = time.time() - time_window
                orders = [o for o in orders if o.timestamp > cutoff]
            
            if not orders:
                return {'message': 'No orders found'}
            
            # Calculate statistics
            total_orders = len(orders)
            filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
            rejected_orders = [o for o in orders if o.status == OrderStatus.REJECTED]
            
            fill_rate = len(filled_orders) / total_orders if total_orders > 0 else 0
            rejection_rate = len(rejected_orders) / total_orders if total_orders > 0 else 0
            
            # Average metrics for filled orders
            if filled_orders:
                avg_fill_time = np.mean([
                    o.last_update_time - o.timestamp 
                    for o in filled_orders
                ])
                avg_slippage = np.mean([o.slippage for o in filled_orders])
            else:
                avg_fill_time = 0
                avg_slippage = 0
            
            return {
                'total_orders': total_orders,
                'filled_orders': len(filled_orders),
                'rejected_orders': len(rejected_orders),
                'fill_rate': fill_rate,
                'rejection_rate': rejection_rate,
                'avg_fill_time': avg_fill_time,
                'avg_slippage': avg_slippage,
                'active_orders': len(self.active_orders),
                'symbols_traded': len(set(o.symbol for o in orders))
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {'error': str(e)}
    
    def cleanup_stale_orders(self, max_age: int = 3600) -> int:
        """
        Clean up stale orders.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            Number of orders cleaned up
        """
        try:
            current_time = time.time()
            stale_orders = []
            
            for order_id, order in self.active_orders.items():
                if order.age_seconds > max_age and not order.is_active:
                    stale_orders.append(order)
            
            for order in stale_orders:
                self.archive_order(order)
            
            logger.info(f"Cleaned up {len(stale_orders)} stale orders")
            return len(stale_orders)
            
        except Exception as e:
            logger.error(f"Error cleaning up stale orders: {e}")
            return 0
