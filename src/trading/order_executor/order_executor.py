"""
File: order_executor.py
Modified: 2025-07-18
Changes Summary:
- Modularized into separate components for better maintainability
- Main order execution coordination remains here
- Components moved to modules/ directory
"""

"""
Order Executor Module

Advanced order execution with smart routing and slippage control.
This module handles order placement, execution monitoring, and
advanced execution algorithms like TWAP and iceberg orders.

Classes:
    OrderExecutor: Main order execution engine
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
import traceback

# Import modularized components
from .modules.order_types import OrderStatus, Order
from .modules.order_validator import OrderValidator
from .modules.execution_algorithms import ExecutionAlgorithms
from .modules.order_tracker import OrderTracker
from .modules.slippage_controller import SlippageController
from .modules.execution_analytics import ExecutionAnalytics

from ...exchange.hyperliquid_client import HyperliquidClient
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class OrderExecutor:
    """
    Advanced order execution with smart routing and slippage control.
    
    This class handles order execution with features including:
    - Smart order routing
    - Slippage protection
    - Large order handling (TWAP, iceberg)
    - Post-only order optimization
    - Execution analytics
    
    Attributes:
        client: Exchange client interface
        active_orders: Currently active orders
        order_history: Historical order records
        params: Execution parameters
        execution_stats: Performance statistics
    """
    
    def __init__(self, exchange_client: HyperliquidClient) -> None:
        """
        Initialize the Order Executor with error handling.
        
        Args:
            exchange_client: Exchange client for order placement
        """
        try:
            # Validate client
            if not exchange_client:
                raise ValueError("Exchange client is required")
                
            self.client = exchange_client
            
            # Initialize components
            self.validator = OrderValidator()
            self.execution_algos = ExecutionAlgorithms(exchange_client)
            self.order_tracker = OrderTracker()
            self.slippage_controller = SlippageController()
            self.analytics = ExecutionAnalytics()
            
            # Execution parameters
            self.params: Dict[str, Any] = {
                'max_slippage': 0.002,  # 0.2% max slippage
                'fill_timeout': 30,  # 30 seconds timeout
                'retry_attempts': 3,
                'chunk_size': 10000,  # $10k per chunk for large orders
                'twap_intervals': 5,  # Number of TWAP intervals
                'aggressive_fill_threshold': 0.8,  # 80% fill rate to become aggressive
                'latency_threshold': 100,  # 100ms latency threshold
                'post_only_retry': 5,  # Retry post-only orders 5 times
                'min_order_size': 1,  # Minimum order size
                'max_order_size': 1000000  # Maximum order size
            }
            
            # Validate parameters
            for key, value in self.params.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    logger.warning(f"Invalid parameter {key}: {value}, using default")
                    self.params[key] = 1
            
            logger.info("OrderExecutor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OrderExecutor: {e}")
            raise
        
    async def place_order(
        self, 
        symbol: str, 
        side: str, 
        size: float, 
        order_type: str = 'limit', 
        price: Optional[float] = None,
        time_in_force: str = 'GTC', 
        post_only: bool = False,
        reduce_only: bool = False, 
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Place order with smart execution logic and comprehensive error handling.
        
        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            size: Order size
            order_type: Order type (market/limit)
            price: Limit price (required for limit orders)
            time_in_force: Time in force (GTC/IOC/FOK)
            post_only: Post-only flag
            reduce_only: Reduce-only flag
            metadata: Additional order metadata
            
        Returns:
            Order execution result dictionary
        """
        try:
            # Validate order
            validation_result = self.validator.validate_order(
                symbol, side, size, order_type, price, self.params
            )
            if not validation_result[0]:
                return {'status': 'rejected', 'reason': validation_result[1]}
            
            # Check if large order needs splitting
            if self._is_large_order(size, price):
                return await self.execution_algos.execute_large_order(
                    symbol, side, size, order_type, price, 
                    time_in_force, post_only, reduce_only, metadata
                )
            
            # Execute single order
            return await self._execute_single_order(
                symbol, side, size, order_type, price,
                time_in_force, post_only, reduce_only, metadata
            )
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            logger.error(traceback.format_exc())
            self.analytics.record_error('order_placement', str(e))
            return {'status': 'error', 'error': str(e)}
    
    async def place_order_async(self, **kwargs) -> Dict[str, Any]:
        """
        Async version of place_order for concurrent execution with error handling.
        
        Args:
            **kwargs: Same arguments as place_order
            
        Returns:
            Order execution result
        """
        try:
            return await self.place_order(**kwargs)
        except Exception as e:
            logger.error(f"Error in async order placement: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _is_large_order(self, size: float, price: Optional[float]) -> bool:
        """
        Check if order is large and needs special handling with error handling.
        
        Args:
            size: Order size
            price: Order price
            
        Returns:
            True if order is large, False otherwise
        """
        try:
            if price and isinstance(price, (int, float)) and price > 0:
                order_value = size * price
                return order_value > self.params['chunk_size'] * 3
            return False
        except Exception as e:
            logger.error(f"Error checking if order is large: {e}")
            return False
    
    async def _execute_single_order(
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
    ) -> Dict[str, Any]:
        """
        Execute a single order with comprehensive error handling.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            size: Order size
            order_type: Order type
            price: Limit price
            time_in_force: Time in force
            post_only: Post-only flag
            reduce_only: Reduce-only flag
            metadata: Order metadata
            
        Returns:
            Execution result dictionary
        """
        start_time = time.time()
        
        try:
            # Create order
            order = self.order_tracker.create_order(
                symbol, side, size, order_type, price,
                time_in_force, post_only, reduce_only, metadata
            )
            
            # Submit order to exchange
            response = None
            for attempt in range(self.params['retry_attempts']):
                try:
                    if order_type == 'market':
                        response = await self._submit_market_order(order)
                    else:
                        response = await self._submit_limit_order(order)
                    
                    if response and response.get('status') == 'success':
                        break
                        
                except Exception as e:
                    logger.warning(f"Order submission attempt {attempt + 1} failed: {e}")
                    if attempt < self.params['retry_attempts'] - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            
            # Handle response
            if response and response.get('status') == 'success':
                self.order_tracker.update_order_status(
                    order.order_id, OrderStatus.SUBMITTED
                )
                
                # Wait for fill or timeout
                fill_result = await self._wait_for_fill(order)
                
                execution_time = time.time() - start_time
                
                # Update analytics
                self.analytics.update_execution_stats(order, execution_time)
                
                return {
                    'status': 'filled' if order.status == OrderStatus.FILLED else 'partial',
                    'order_id': order.order_id,
                    'fill_price': order.avg_fill_price,
                    'filled_size': order.filled_size,
                    'remaining_size': order.size - order.filled_size,
                    'execution_time': execution_time,
                    'slippage': self.slippage_controller.calculate_slippage(order, price),
                    'fees': response.get('fees', 0)
                }
            else:
                self.order_tracker.update_order_status(
                    order.order_id, OrderStatus.REJECTED
                )
                self.analytics.record_rejection(order)
                return {
                    'status': 'rejected',
                    'reason': response.get('error', 'unknown') if response else 'no response',
                    'order_id': order.order_id
                }
                
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            logger.error(traceback.format_exc())
            if 'order' in locals():
                self.order_tracker.update_order_status(
                    order.order_id, OrderStatus.REJECTED
                )
            self.analytics.record_error('order_execution', str(e))
            return {
                'status': 'error',
                'error': str(e),
                'order_id': order.order_id if 'order' in locals() else 'unknown'
            }
        finally:
            # Move to history
            if 'order' in locals():
                self.order_tracker.archive_order(order)
    
    async def _submit_market_order(self, order: Order) -> Dict[str, Any]:
        """
        Submit market order to exchange with slippage protection.
        
        Args:
            order: Order object
            
        Returns:
            Exchange response
        """
        try:
            return await self.slippage_controller.execute_with_slippage_protection(
                self.client, order, self.params['max_slippage']
            )
        except Exception as e:
            logger.error(f"Error submitting market order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _submit_limit_order(self, order: Order) -> Dict[str, Any]:
        """
        Submit limit order to exchange with post-only handling.
        
        Args:
            order: Order object
            
        Returns:
            Exchange response
        """
        try:
            return await self.execution_algos.submit_limit_order_with_retry(
                order, self.params['post_only_retry']
            )
        except Exception as e:
            logger.error(f"Error submitting limit order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _wait_for_fill(self, order: Order) -> Dict[str, Any]:
        """
        Wait for order to fill or timeout with error handling.
        
        Args:
            order: Order object
            
        Returns:
            Fill status dictionary
        """
        start_time = time.time()
        check_interval = 0.1  # 100ms
        
        try:
            while time.time() - start_time < self.params['fill_timeout']:
                try:
                    # Check order status
                    status = await self.client.get_order_status(order.order_id)
                    
                    # Validate status response
                    if not status or not isinstance(status, dict):
                        logger.warning(f"Invalid order status response for {order.order_id}")
                        await asyncio.sleep(check_interval)
                        continue
                    
                    # Update order based on status
                    fill_result = self.order_tracker.process_order_update(order, status)
                    
                    if fill_result['complete']:
                        return status
                    
                    # Check if we should become more aggressive
                    if fill_result['should_replace']:
                        await self.execution_algos.replace_with_aggressive_order(
                            order, self.params['aggressive_fill_threshold']
                        )
                    
                except Exception as e:
                    logger.warning(f"Error checking order status: {e}")
                    
                await asyncio.sleep(check_interval)
            
            # Timeout reached
            return self.order_tracker.handle_timeout(order, self.client)
            
        except Exception as e:
            logger.error(f"Error waiting for fill: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'filled_size': order.filled_size,
                'avg_fill_price': order.avg_fill_price
            }
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order with error handling.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            Cancellation result
        """
        try:
            # Validate inputs
            if not symbol or not order_id:
                return {'status': 'error', 'error': 'Invalid parameters'}
                
            result = await self.client.cancel_order(order_id)
            
            # Update order status
            self.order_tracker.update_order_status(order_id, OrderStatus.CANCELLED)
                
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def cancel_all_orders(
        self, 
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Cancel all orders for a symbol or all symbols with error handling.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of cancellation results
        """
        results = []
        
        try:
            active_orders = self.order_tracker.get_active_orders(symbol)
            
            for order in active_orders:
                try:
                    result = await self.cancel_order(order.symbol, order.order_id)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error cancelling order {order.order_id}: {e}")
                    results.append({
                        'status': 'error', 
                        'order_id': order.order_id, 
                        'error': str(e)
                    })
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            
        return results
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """
        Get execution performance analytics with error handling.
        
        Returns:
            Dictionary containing execution metrics
        """
        try:
            return self.analytics.get_analytics()
        except Exception as e:
            logger.error(f"Error getting execution analytics: {e}")
            return {'error': str(e)}

"""
MODULARIZATION_SUMMARY:
- Original file: 900+ lines
- Main file: ~300 lines (core coordination)
- Modules created:
  - order_types.py: Order data structures and enums
  - order_validator.py: Order validation logic
  - execution_algorithms.py: TWAP, Iceberg, and other algorithms
  - order_tracker.py: Order tracking and state management
  - slippage_controller.py: Slippage protection mechanisms
  - execution_analytics.py: Performance tracking and analytics
- Benefits:
  - Clearer separation of concerns
  - Easier testing of execution algorithms
  - Better order tracking
  - Modular analytics
"""
