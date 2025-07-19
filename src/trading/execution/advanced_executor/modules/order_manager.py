"""
Order management module for advanced order execution.
Handles child order placement, retries, and tracking.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages child order placement and tracking"""
    
    def __init__(self, executor):
        self.executor = executor
        
    async def place_child_order_with_retries(self,
                                           symbol: str,
                                           side: str,
                                           size: float,
                                           slice_number: int = 1,
                                           execution_id: str = None,
                                           price: Optional[float] = None) -> Optional[Dict]:
        """
        Place individual child order with retry logic
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            slice_number: Slice number for tracking
            execution_id: Parent execution ID
            price: Limit price (None for market)
            
        Returns:
            Order result dictionary or None if failed
        """
        for retry in range(self.executor.max_slice_retries):
            try:
                child_order = await self.place_child_order(
                    symbol=symbol,
                    side=side,
                    size=size,
                    slice_number=slice_number,
                    execution_id=execution_id,
                    price=price
                )
                
                if child_order and child_order.get('status') != 'error':
                    return child_order
                    
                # Exponential backoff for retries
                if retry < self.executor.max_slice_retries - 1:
                    wait_time = (2 ** retry) * 2
                    logger.warning(f"Retrying child order in {wait_time}s (attempt {retry + 2}/{self.executor.max_slice_retries})")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Error in child order retry {retry + 1}: {e}")
                if retry < self.executor.max_slice_retries - 1:
                    await asyncio.sleep((2 ** retry) * 2)
                    
        return None
    
    async def place_child_order(self,
                              symbol: str,
                              side: str,
                              size: float,
                              slice_number: int = 1,
                              execution_id: str = None,
                              price: Optional[float] = None) -> Optional[Dict]:
        """
        Place individual child order as part of larger execution
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            slice_number: Slice number for tracking
            execution_id: Parent execution ID
            price: Limit price (None for market)
            
        Returns:
            Order result dictionary
        """
        try:
            # Validate order parameters
            if size < self.executor.min_order_size:
                logger.warning(f"Order size {size} below minimum {self.executor.min_order_size}")
                return None
            
            # Get current market price if no price specified
            if price is None:
                price = await self._get_market_price(symbol, side)
                if not price:
                    return None
            
            # Place order through exchange
            order_result = await self._place_exchange_order(
                symbol, side, size, price, execution_id, slice_number
            )
            
            if not order_result or 'id' not in order_result:
                logger.error("Invalid order result from exchange")
                return {'status': 'error', 'message': 'Invalid order result'}
            
            # Wait for fill with timeout
            fill_result = await self._wait_for_fill(order_result, symbol)
            
            logger.debug(f"Child order executed: {fill_result}")
            return fill_result
            
        except Exception as e:
            logger.error(f"Error placing child order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _get_market_price(self, symbol: str, side: str) -> Optional[float]:
        """Get current market price for order placement"""
        try:
            ticker = await self.executor.exchange.get_ticker(symbol)
            if not ticker:
                raise ValueError("Failed to get ticker data")
                
            if side == 'buy':
                price = ticker.get('ask', ticker.get('last', 0))
            else:
                price = ticker.get('bid', ticker.get('last', 0))
                
            # Validate market price
            if not price or price <= 0:
                raise ValueError(f"Invalid market price: {price}")
                
            return price
            
        except Exception as e:
            logger.error(f"Error getting market price: {e}")
            return None
    
    async def _place_exchange_order(self, symbol: str, side: str, size: float,
                                  price: float, execution_id: str,
                                  slice_number: int) -> Dict:
        """Place order through exchange API"""
        return await self.executor.exchange.place_order(
            symbol=symbol,
            side=side,
            amount=size,
            order_type='limit' if price else 'market',
            price=price,
            params={
                'execution_id': execution_id,
                'slice_number': slice_number
            }
        )
    
    async def _wait_for_fill(self, order_result: Dict, symbol: str) -> Dict:
        """Wait for order fill with timeout"""
        fill_timeout = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < fill_timeout:
            try:
                # Check order status
                order_status = await self.executor.exchange.get_order_status(
                    order_result['id'], symbol
                )
                
                if order_status and order_status.get('status') in ['filled', 'partial', 'cancelled']:
                    break
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error checking order status: {e}")
                await asyncio.sleep(1)
        
        # Get final order details
        return await self._get_final_order_details(order_result, symbol)
    
    async def _get_final_order_details(self, order_result: Dict, symbol: str) -> Dict:
        """Get final order details after execution"""
        try:
            final_order = await self.executor.exchange.get_order_status(
                order_result['id'], symbol
            )
            
            if final_order:
                return {
                    'order_id': final_order.get('id', order_result['id']),
                    'symbol': symbol,
                    'side': final_order.get('side'),
                    'size': final_order.get('amount'),
                    'price': final_order.get('price'),
                    'fill_price': final_order.get('average', final_order.get('price')),
                    'filled_size': final_order.get('filled', 0),
                    'status': final_order.get('status', 'unknown'),
                    'timestamp': datetime.now(),
                    'execution_id': order_result.get('params', {}).get('execution_id'),
                    'slice_number': order_result.get('params', {}).get('slice_number')
                }
            else:
                # Fallback result
                return {
                    'order_id': order_result['id'],
                    'symbol': symbol,
                    'side': order_result.get('side'),
                    'size': order_result.get('amount'),
                    'price': order_result.get('price'),
                    'fill_price': order_result.get('price'),
                    'filled_size': 0,
                    'status': 'unknown',
                    'timestamp': datetime.now(),
                    'execution_id': order_result.get('params', {}).get('execution_id'),
                    'slice_number': order_result.get('params', {}).get('slice_number')
                }
                
        except Exception as e:
            logger.error(f"Error getting final order details: {e}")
            return {
                'order_id': order_result.get('id', 'unknown'),
                'status': 'error',
                'error': str(e)
            }
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel active execution algorithm
        
        Args:
            execution_id: Execution ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            if execution_id in self.executor.active_orders:
                execution = self.executor.active_orders[execution_id]
                execution['status'] = 'cancelled'
                execution['end_time'] = datetime.now()
                
                # Move to history
                self.executor._add_to_history(execution)
                del self.executor.active_orders[execution_id]
                
                logger.info(f"Execution {execution_id} cancelled")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling execution: {e}")
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """
        Get status of active or completed execution
        
        Args:
            execution_id: Execution ID to check
            
        Returns:
            Execution status dictionary or None
        """
        try:
            # Check active orders first
            if execution_id in self.executor.active_orders:
                return self.executor.active_orders[execution_id].copy()
            
            # Check history
            for execution in self.executor.execution_history:
                if execution.get('execution_id') == execution_id:
                    return execution.copy()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting execution status: {e}")
            return None
    
    def get_active_executions(self) -> List[Dict]:
        """
        Get list of all active executions
        
        Returns:
            List of active execution summaries
        """
        try:
            return [
                {
                    'execution_id': exec_id,
                    'symbol': exec_data.get('symbol'),
                    'side': exec_data.get('side'),
                    'total_size': exec_data.get('total_size'),
                    'filled': exec_data.get('total_filled', 0),
                    'status': exec_data.get('status'),
                    'start_time': exec_data.get('start_time')
                }
                for exec_id, exec_data in self.executor.active_orders.items()
            ]
        except Exception as e:
            logger.error(f"Error getting active executions: {e}")
            return []
    
    async def emergency_cancel_all(self) -> Dict[str, int]:
        """
        Emergency cancel all active executions
        
        Returns:
            Dictionary with counts of cancelled and failed cancellations
        """
        logger.warning("Emergency cancel initiated for all active executions")
        
        results = {'cancelled': 0, 'failed': 0}
        
        # Copy keys to avoid modification during iteration
        execution_ids = list(self.executor.active_orders.keys())
        
        for execution_id in execution_ids:
            try:
                if self.cancel_execution(execution_id):
                    results['cancelled'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                logger.error(f"Error cancelling {execution_id}: {e}")
                results['failed'] += 1
        
        return results
