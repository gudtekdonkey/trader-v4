import asyncio
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from ..exchange.hyperliquid_client import HyperliquidClient
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
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
    metadata: Dict

class OrderExecutor:
    """Advanced order execution with smart routing and slippage control"""
    
    def __init__(self, exchange_client: HyperliquidClient):
        self.client = exchange_client
        self.active_orders = {}
        self.order_history = []
        
        # Execution parameters
        self.params = {
            'max_slippage': 0.002,  # 0.2% max slippage
            'fill_timeout': 30,  # 30 seconds timeout
            'retry_attempts': 3,
            'chunk_size': 10000,  # $10k per chunk for large orders
            'twap_intervals': 5,  # Number of TWAP intervals
            'aggressive_fill_threshold': 0.8,  # 80% fill rate to become aggressive
            'latency_threshold': 100,  # 100ms latency threshold
            'post_only_retry': 5  # Retry post-only orders 5 times
        }
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'avg_slippage': 0,
            'avg_fill_time': 0,
            'total_fees': 0
        }
        
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = 'limit', price: Optional[float] = None,
                         time_in_force: str = 'GTC', post_only: bool = False,
                         reduce_only: bool = False, metadata: Dict = None) -> Dict:
        """Place order with smart execution logic"""
        
        # Validate order
        if not self._validate_order(symbol, side, size, order_type, price):
            return {'status': 'rejected', 'reason': 'validation_failed'}
        
        # Check if large order needs splitting
        if self._is_large_order(size, price):
            return await self._execute_large_order(
                symbol, side, size, order_type, price, 
                time_in_force, post_only, reduce_only, metadata
            )
        
        # Execute single order
        return await self._execute_single_order(
            symbol, side, size, order_type, price,
            time_in_force, post_only, reduce_only, metadata
        )
    
    async def place_order_async(self, **kwargs) -> Dict:
        """Async version of place_order for concurrent execution"""
        return await self.place_order(**kwargs)
    
    def _validate_order(self, symbol: str, side: str, size: float, 
                       order_type: str, price: Optional[float]) -> bool:
        """Validate order parameters"""
        if size <= 0:
            logger.error("Invalid order size")
            return False
        
        if order_type == 'limit' and price is None:
            logger.error("Limit order requires price")
            return False
        
        if side not in ['buy', 'sell']:
            logger.error("Invalid order side")
            return False
        
        return True
    
    def _is_large_order(self, size: float, price: Optional[float]) -> bool:
        """Check if order is large and needs special handling"""
        if price:
            order_value = size * price
            return order_value > self.params['chunk_size'] * 3
        return False
    
    async def _execute_single_order(self, symbol: str, side: str, size: float,
                                  order_type: str, price: Optional[float],
                                  time_in_force: str, post_only: bool,
                                  reduce_only: bool, metadata: Dict) -> Dict:
        """Execute a single order"""
        start_time = time.time()
        
        # Prepare order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            size=size,
            order_type=order_type,
            price=price,
            status=OrderStatus.PENDING,
            filled_size=0,
            avg_fill_price=0,
            timestamp=start_time,
            time_in_force=time_in_force,
            post_only=post_only,
            reduce_only=reduce_only,
            metadata=metadata or {}
        )
        
        # Track order
        self.active_orders[order.order_id] = order
        
        try:
            # Submit order to exchange
            if order_type == 'market':
                response = await self._submit_market_order(order)
            else:
                response = await self._submit_limit_order(order)
            
            # Handle response
            if response['status'] == 'success':
                order.status = OrderStatus.SUBMITTED
                
                # Wait for fill or timeout
                fill_result = await self._wait_for_fill(order)
                
                execution_time = time.time() - start_time
                
                # Update stats
                self._update_execution_stats(order, execution_time)
                
                return {
                    'status': 'filled' if order.status == OrderStatus.FILLED else 'partial',
                    'order_id': order.order_id,
                    'fill_price': order.avg_fill_price,
                    'filled_size': order.filled_size,
                    'remaining_size': order.size - order.filled_size,
                    'execution_time': execution_time,
                    'slippage': self._calculate_slippage(order, price),
                    'fees': response.get('fees', 0)
                }
            else:
                order.status = OrderStatus.REJECTED
                return {
                    'status': 'rejected',
                    'reason': response.get('error', 'unknown'),
                    'order_id': order.order_id
                }
                
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            order.status = OrderStatus.REJECTED
            return {
                'status': 'error',
                'error': str(e),
                'order_id': order.order_id
            }
        finally:
            # Move to history
            self.order_history.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
    
    async def _submit_market_order(self, order: Order) -> Dict:
        """Submit market order to exchange"""
        # Get current market price for slippage protection
        ticker = await self.client.get_ticker(order.symbol)
        
        if order.side == 'buy':
            expected_price = ticker['ask']
            max_price = expected_price * (1 + self.params['max_slippage'])
        else:
            expected_price = ticker['bid']
            max_price = expected_price * (1 - self.params['max_slippage'])
        
        # Submit as aggressive limit order
        return await self.client.place_order(
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            order_type='limit',
            price=max_price,
            time_in_force='IOC',  # Immediate or cancel
            reduce_only=order.reduce_only
        )
    
    async def _submit_limit_order(self, order: Order) -> Dict:
        """Submit limit order to exchange"""
        attempts = 0
        
        while attempts < self.params['post_only_retry'] and order.post_only:
            response = await self.client.place_order(
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                order_type='limit',
                price=order.price,
                time_in_force=order.time_in_force,
                post_only=order.post_only,
                reduce_only=order.reduce_only
            )
            
            if response['status'] == 'success' or not response.get('post_only_reject'):
                return response
            
            # Adjust price for post-only
            attempts += 1
            if order.side == 'buy':
                order.price *= 0.9999  # Slightly lower bid
            else:
                order.price *= 1.0001  # Slightly higher ask
            
            await asyncio.sleep(0.1)
        
        # Final attempt without post-only if needed
        if order.post_only and attempts >= self.params['post_only_retry']:
            order.post_only = False
            
        return await self.client.place_order(
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            order_type='limit',
            price=order.price,
            time_in_force=order.time_in_force,
            post_only=order.post_only,
            reduce_only=order.reduce_only
        )
    
    async def _wait_for_fill(self, order: Order) -> Dict:
        """Wait for order to fill or timeout"""
        start_time = time.time()
        check_interval = 0.1  # 100ms
        
        while time.time() - start_time < self.params['fill_timeout']:
            # Check order status
            status = await self.client.get_order_status(order.order_id)
            
            if status['status'] == 'filled':
                order.status = OrderStatus.FILLED
                order.filled_size = status['filled_size']
                order.avg_fill_price = status['avg_fill_price']
                return status
            
            elif status['status'] == 'partial':
                order.status = OrderStatus.PARTIAL
                order.filled_size = status['filled_size']
                order.avg_fill_price = status['avg_fill_price']
                
                # Check if we should become more aggressive
                fill_rate = order.filled_size / order.size
                if fill_rate < self.params['aggressive_fill_threshold']:
                    # Cancel and replace with more aggressive order
                    await self._replace_with_aggressive_order(order)
            
            elif status['status'] in ['cancelled', 'rejected', 'expired']:
                order.status = OrderStatus(status['status'].upper())
                return status
            
            await asyncio.sleep(check_interval)
        
        # Timeout reached
        if order.filled_size > 0:
            order.status = OrderStatus.PARTIAL
        else:
            order.status = OrderStatus.EXPIRED
            
        # Cancel remaining
        await self.client.cancel_order(order.order_id)
        
        return {
            'status': order.status.value,
            'filled_size': order.filled_size,
            'avg_fill_price': order.avg_fill_price
        }
    
    async def _replace_with_aggressive_order(self, order: Order):
        """Replace order with more aggressive pricing"""
        # Cancel existing order
        await self.client.cancel_order(order.order_id)
        
        # Get current market
        ticker = await self.client.get_ticker(order.symbol)
        
        # New aggressive price
        if order.side == 'buy':
            new_price = ticker['ask'] * 1.001  # Cross the spread slightly
        else:
            new_price = ticker['bid'] * 0.999
        
        # Submit new order for remaining size
        remaining_size = order.size - order.filled_size
        
        new_response = await self.client.place_order(
            symbol=order.symbol,
            side=order.side,
            size=remaining_size,
            order_type='limit',
            price=new_price,
            time_in_force='IOC'
        )
        
        # Update order if successful
        if new_response['status'] == 'success':
            order.order_id = new_response['order_id']
    
    async def _execute_large_order(self, symbol: str, side: str, size: float,
                                 order_type: str, price: Optional[float],
                                 time_in_force: str, post_only: bool,
                                 reduce_only: bool, metadata: Dict) -> Dict:
        """Execute large order using TWAP or iceberg"""
        logger.info(f"Executing large order: {size} {symbol}")
        
        # Determine execution strategy
        if metadata and metadata.get('execution_algo') == 'twap':
            return await self._execute_twap(
                symbol, side, size, price, time_in_force, 
                post_only, reduce_only, metadata
            )
        else:
            return await self._execute_iceberg(
                symbol, side, size, order_type, price,
                time_in_force, post_only, reduce_only, metadata
            )
    
    async def _execute_twap(self, symbol: str, side: str, total_size: float,
                          price: Optional[float], time_in_force: str,
                          post_only: bool, reduce_only: bool, metadata: Dict) -> Dict:
        """Execute order using Time-Weighted Average Price algorithm"""
        intervals = self.params['twap_intervals']
        interval_size = total_size / intervals
        interval_delay = metadata.get('twap_duration', 300) / intervals  # Default 5 minutes
        
        results = []
        total_filled = 0
        total_cost = 0
        
        for i in range(intervals):
            # Adjust size for last interval
            if i == intervals - 1:
                current_size = total_size - total_filled
            else:
                current_size = interval_size
            
            # Execute chunk
            result = await self._execute_single_order(
                symbol, side, current_size, 'limit', price,
                'IOC', post_only, reduce_only, 
                {**metadata, 'chunk': i+1, 'total_chunks': intervals}
            )
            
            results.append(result)
            
            if result['status'] in ['filled', 'partial']:
                total_filled += result['filled_size']
                total_cost += result['filled_size'] * result['fill_price']
            
            # Wait before next interval (except last)
            if i < intervals - 1:
                await asyncio.sleep(interval_delay)
        
        # Aggregate results
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        
        return {
            'status': 'filled' if total_filled == total_size else 'partial',
            'execution_type': 'twap',
            'total_size': total_size,
            'filled_size': total_filled,
            'avg_price': avg_price,
            'chunks': results,
            'fill_rate': total_filled / total_size
        }
    
    async def _execute_iceberg(self, symbol: str, side: str, total_size: float,
                             order_type: str, price: Optional[float],
                             time_in_force: str, post_only: bool,
                             reduce_only: bool, metadata: Dict) -> Dict:
        """Execute order using iceberg (hidden size) strategy"""
        visible_size = min(self.params['chunk_size'] / price if price else 100, total_size * 0.1)
        
        total_filled = 0
        total_cost = 0
        chunks = []
        
        while total_filled < total_size:
            # Calculate chunk size
            remaining = total_size - total_filled
            chunk_size = min(visible_size, remaining)
            
            # Execute chunk
            result = await self._execute_single_order(
                symbol, side, chunk_size, order_type, price,
                time_in_force, post_only, reduce_only,
                {**metadata, 'iceberg': True}
            )
            
            chunks.append(result)
            
            if result['status'] in ['filled', 'partial']:
                total_filled += result['filled_size']
                total_cost += result['filled_size'] * result['fill_price']
                
                # Update price for next chunk if needed
                if order_type == 'limit' and price:
                    # Adjust price based on market movement
                    ticker = await self.client.get_ticker(symbol)
                    if side == 'buy' and ticker['ask'] < price:
                        price = ticker['ask']
                    elif side == 'sell' and ticker['bid'] > price:
                        price = ticker['bid']
            else:
                # Failed chunk, stop execution
                break
            
            # Small delay between chunks
            await asyncio.sleep(0.5)
        
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        
        return {
            'status': 'filled' if total_filled == total_size else 'partial',
            'execution_type': 'iceberg',
            'total_size': total_size,
            'filled_size': total_filled,
            'avg_price': avg_price,
            'chunks': chunks,
            'fill_rate': total_filled / total_size
        }
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order"""
        try:
            result = await self.client.cancel_order(order_id)
            
            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED
                
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Cancel all orders for a symbol or all symbols"""
        results = []
        
        for order_id, order in list(self.active_orders.items()):
            if symbol is None or order.symbol == symbol:
                result = await self.cancel_order(order.symbol, order_id)
                results.append(result)
        
        return results
    
    def _calculate_slippage(self, order: Order, expected_price: Optional[float]) -> float:
        """Calculate slippage from expected price"""
        if not expected_price or order.avg_fill_price == 0:
            return 0
        
        if order.side == 'buy':
            slippage = (order.avg_fill_price - expected_price) / expected_price
        else:
            slippage = (expected_price - order.avg_fill_price) / expected_price
        
        return slippage
    
    def _update_execution_stats(self, order: Order, execution_time: float):
        """Update execution statistics"""
        self.execution_stats['total_orders'] += 1
        
        if order.status == OrderStatus.FILLED:
            self.execution_stats['filled_orders'] += 1
        elif order.status == OrderStatus.REJECTED:
            self.execution_stats['rejected_orders'] += 1
        
        # Update average slippage
        if order.metadata.get('expected_price'):
            slippage = self._calculate_slippage(order, order.metadata['expected_price'])
            
            # Running average
            n = self.execution_stats['filled_orders']
            prev_avg = self.execution_stats['avg_slippage']
            self.execution_stats['avg_slippage'] = (prev_avg * (n-1) + slippage) / n
        
        # Update average fill time
        n = self.execution_stats['filled_orders']
        prev_avg = self.execution_stats['avg_fill_time']
        self.execution_stats['avg_fill_time'] = (prev_avg * (n-1) + execution_time) / n
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"ORD_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
    
    def get_execution_analytics(self) -> Dict:
        """Get execution performance analytics"""
        fill_rate = (self.execution_stats['filled_orders'] / 
                    self.execution_stats['total_orders'] 
                    if self.execution_stats['total_orders'] > 0 else 0)
        
        return {
            'fill_rate': fill_rate,
            'avg_slippage_bps': self.execution_stats['avg_slippage'] * 10000,  # Basis points
            'avg_fill_time_ms': self.execution_stats['avg_fill_time'] * 1000,
            'total_orders': self.execution_stats['total_orders'],
            'active_orders': len(self.active_orders),
            'rejection_rate': (self.execution_stats['rejected_orders'] / 
                             self.execution_stats['total_orders']
                             if self.execution_stats['total_orders'] > 0 else 0)
        }
