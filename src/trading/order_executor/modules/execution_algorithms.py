"""
Execution Algorithms Module

Implements advanced order execution algorithms including TWAP,
VWAP, Iceberg orders, and aggressive fill strategies.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
import numpy as np

from .order_types import Order, OrderStatus

logger = logging.getLogger(__name__)


class ExecutionAlgorithms:
    """
    Advanced execution algorithms for optimal order execution.
    
    Implements various execution strategies to minimize market impact
    and achieve best execution for large orders.
    """
    
    def __init__(self, exchange_client):
        """
        Initialize execution algorithms.
        
        Args:
            exchange_client: Exchange client for order placement
        """
        self.client = exchange_client
        self.active_algos = {}
    
    async def execute_large_order(
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
        Execute large order using appropriate algorithm.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            size: Total order size
            order_type: Order type
            price: Limit price
            time_in_force: Time in force
            post_only: Post-only flag
            reduce_only: Reduce-only flag
            metadata: Additional metadata
            
        Returns:
            Execution result
        """
        try:
            # Determine best execution algorithm
            algo_type = self._select_algorithm(size, order_type, metadata)
            
            logger.info(f"Executing large order using {algo_type} algorithm")
            
            if algo_type == 'TWAP':
                return await self.execute_twap(
                    symbol, side, size, price, 
                    metadata.get('twap_duration', 300),  # 5 min default
                    metadata.get('twap_intervals', 5)
                )
            elif algo_type == 'ICEBERG':
                return await self.execute_iceberg(
                    symbol, side, size, price,
                    metadata.get('show_size', size * 0.1),  # Show 10% default
                    post_only, reduce_only
                )
            elif algo_type == 'AGGRESSIVE':
                return await self.execute_aggressive(
                    symbol, side, size, order_type, price
                )
            else:
                # Default to TWAP
                return await self.execute_twap(
                    symbol, side, size, price, 300, 5
                )
                
        except Exception as e:
            logger.error(f"Error executing large order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _select_algorithm(
        self,
        size: float,
        order_type: str,
        metadata: Optional[Dict]
    ) -> str:
        """Select appropriate execution algorithm."""
        if metadata and 'algo_type' in metadata:
            return metadata['algo_type']
        
        if order_type == 'market':
            return 'AGGRESSIVE'
        else:
            # Use TWAP for very large orders, ICEBERG for medium
            if metadata and metadata.get('order_value', 0) > 100000:
                return 'TWAP'
            else:
                return 'ICEBERG'
    
    async def execute_twap(
        self,
        symbol: str,
        side: str,
        total_size: float,
        price: Optional[float],
        duration: int = 300,
        intervals: int = 5
    ) -> Dict[str, Any]:
        """
        Execute Time-Weighted Average Price (TWAP) order.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            total_size: Total size to execute
            price: Limit price
            duration: Total duration in seconds
            intervals: Number of intervals
            
        Returns:
            Execution result
        """
        try:
            slice_size = total_size / intervals
            interval_duration = duration / intervals
            
            results = []
            total_filled = 0
            total_cost = 0
            
            logger.info(
                f"Starting TWAP: {total_size} {symbol} over {duration}s "
                f"in {intervals} slices of {slice_size}"
            )
            
            for i in range(intervals):
                start_time = time.time()
                
                # Calculate adaptive price for this slice
                slice_price = await self._calculate_slice_price(
                    symbol, side, price, i, intervals
                )
                
                # Execute slice
                result = await self.client.place_order(
                    symbol=symbol,
                    side=side,
                    size=slice_size,
                    order_type='limit' if price else 'market',
                    price=slice_price,
                    time_in_force='IOC'
                )
                
                if result.get('status') == 'success':
                    filled = result.get('filled_size', 0)
                    fill_price = result.get('avg_price', 0)
                    
                    total_filled += filled
                    total_cost += filled * fill_price
                    
                    results.append({
                        'slice': i + 1,
                        'filled': filled,
                        'price': fill_price,
                        'timestamp': time.time()
                    })
                    
                    logger.info(
                        f"TWAP slice {i+1}/{intervals}: "
                        f"Filled {filled}/{slice_size} @ {fill_price}"
                    )
                else:
                    logger.warning(f"TWAP slice {i+1} failed: {result}")
                
                # Wait for next interval (if not last)
                if i < intervals - 1:
                    elapsed = time.time() - start_time
                    wait_time = max(0, interval_duration - elapsed)
                    await asyncio.sleep(wait_time)
            
            # Calculate summary
            avg_price = total_cost / total_filled if total_filled > 0 else 0
            fill_rate = total_filled / total_size
            
            return {
                'status': 'completed',
                'algo': 'TWAP',
                'total_size': total_size,
                'filled_size': total_filled,
                'avg_price': avg_price,
                'fill_rate': fill_rate,
                'slices': results,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Error in TWAP execution: {e}")
            return {
                'status': 'error',
                'algo': 'TWAP',
                'error': str(e),
                'filled_size': total_filled if 'total_filled' in locals() else 0
            }
    
    async def execute_iceberg(
        self,
        symbol: str,
        side: str,
        total_size: float,
        price: float,
        show_size: float,
        post_only: bool = True,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Iceberg order (show only part of total size).
        
        Args:
            symbol: Trading symbol
            side: Trade side
            total_size: Total size to execute
            price: Limit price
            show_size: Visible size in orderbook
            post_only: Post-only flag
            reduce_only: Reduce-only flag
            
        Returns:
            Execution result
        """
        try:
            remaining_size = total_size
            total_filled = 0
            total_cost = 0
            order_count = 0
            
            logger.info(
                f"Starting Iceberg: {total_size} {symbol} @ {price}, "
                f"showing {show_size}"
            )
            
            while remaining_size > 0:
                # Determine slice size
                current_slice = min(show_size, remaining_size)
                
                # Place order
                result = await self.client.place_order(
                    symbol=symbol,
                    side=side,
                    size=current_slice,
                    order_type='limit',
                    price=price,
                    time_in_force='GTC',
                    post_only=post_only,
                    reduce_only=reduce_only
                )
                
                if result.get('status') == 'success':
                    order_id = result.get('order_id')
                    order_count += 1
                    
                    # Monitor until filled or cancelled
                    fill_result = await self._monitor_iceberg_slice(
                        order_id, current_slice
                    )
                    
                    filled = fill_result.get('filled_size', 0)
                    fill_price = fill_result.get('avg_price', price)
                    
                    total_filled += filled
                    total_cost += filled * fill_price
                    remaining_size -= filled
                    
                    logger.info(
                        f"Iceberg slice {order_count}: "
                        f"Filled {filled}/{current_slice} @ {fill_price}, "
                        f"Remaining: {remaining_size}"
                    )
                    
                    # Check if we should stop
                    if filled < current_slice * 0.5:  # Less than 50% filled
                        logger.warning("Iceberg execution slowing, may need to adjust")
                        # Could implement price adjustment logic here
                    
                else:
                    logger.error(f"Failed to place iceberg slice: {result}")
                    break
                
                # Small delay between slices
                await asyncio.sleep(1)
            
            # Calculate summary
            avg_price = total_cost / total_filled if total_filled > 0 else 0
            fill_rate = total_filled / total_size
            
            return {
                'status': 'completed',
                'algo': 'ICEBERG',
                'total_size': total_size,
                'filled_size': total_filled,
                'avg_price': avg_price,
                'fill_rate': fill_rate,
                'order_count': order_count,
                'show_size': show_size
            }
            
        except Exception as e:
            logger.error(f"Error in Iceberg execution: {e}")
            return {
                'status': 'error',
                'algo': 'ICEBERG',
                'error': str(e),
                'filled_size': total_filled if 'total_filled' in locals() else 0
            }
    
    async def execute_aggressive(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float]
    ) -> Dict[str, Any]:
        """
        Execute order aggressively to ensure fill.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            size: Order size
            order_type: Order type
            price: Limit price
            
        Returns:
            Execution result
        """
        try:
            # Get current orderbook
            orderbook = await self.client.get_orderbook(symbol)
            
            if order_type == 'market' or price is None:
                # Calculate sweep price
                sweep_price = self._calculate_sweep_price(
                    orderbook, side, size
                )
                
                # Add buffer for slippage
                if side == 'buy':
                    limit_price = sweep_price * 1.01  # 1% buffer
                else:
                    limit_price = sweep_price * 0.99
            else:
                limit_price = price
            
            # Execute as aggressive limit order
            result = await self.client.place_order(
                symbol=symbol,
                side=side,
                size=size,
                order_type='limit',
                price=limit_price,
                time_in_force='IOC'
            )
            
            if result.get('status') == 'success':
                filled = result.get('filled_size', 0)
                
                # If not fully filled, try again with worse price
                if filled < size:
                    remaining = size - filled
                    
                    if side == 'buy':
                        retry_price = limit_price * 1.02
                    else:
                        retry_price = limit_price * 0.98
                    
                    retry_result = await self.client.place_order(
                        symbol=symbol,
                        side=side,
                        size=remaining,
                        order_type='limit',
                        price=retry_price,
                        time_in_force='IOC'
                    )
                    
                    if retry_result.get('status') == 'success':
                        result['filled_size'] = filled + retry_result.get('filled_size', 0)
                        # Recalculate average price
                        total_cost = (filled * result.get('avg_price', 0) + 
                                    retry_result.get('filled_size', 0) * retry_result.get('avg_price', 0))
                        result['avg_price'] = total_cost / result['filled_size'] if result['filled_size'] > 0 else 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error in aggressive execution: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _calculate_slice_price(
        self,
        symbol: str,
        side: str,
        base_price: Optional[float],
        slice_index: int,
        total_slices: int
    ) -> Optional[float]:
        """Calculate adaptive price for TWAP slice."""
        if base_price is None:
            return None
        
        try:
            # Get current market price
            ticker = await self.client.get_ticker(symbol)
            mid_price = (ticker['bid'] + ticker['ask']) / 2
            
            # Calculate price adjustment based on slice position
            # Earlier slices can be less aggressive
            aggressiveness = slice_index / total_slices
            
            if side == 'buy':
                # Start below base price, move up
                adjusted_price = base_price * (1 - 0.001 * (1 - aggressiveness))
                # But don't go above mid + small buffer
                return min(adjusted_price, mid_price * 1.0001)
            else:
                # Start above base price, move down
                adjusted_price = base_price * (1 + 0.001 * (1 - aggressiveness))
                # But don't go below mid - small buffer
                return max(adjusted_price, mid_price * 0.9999)
                
        except Exception as e:
            logger.warning(f"Error calculating slice price: {e}")
            return base_price
    
    async def _monitor_iceberg_slice(
        self,
        order_id: str,
        target_size: float,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Monitor iceberg order slice until filled or timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status = await self.client.get_order_status(order_id)
                
                if status.get('status') in ['filled', 'cancelled', 'rejected']:
                    return status
                
                # Check if partially filled and stalled
                if status.get('status') == 'partial':
                    filled = status.get('filled_size', 0)
                    if filled > target_size * 0.8:  # 80% filled
                        # Cancel remainder and return
                        await self.client.cancel_order(order_id)
                        return status
                
            except Exception as e:
                logger.warning(f"Error monitoring order {order_id}: {e}")
            
            await asyncio.sleep(0.5)
        
        # Timeout - cancel order
        try:
            await self.client.cancel_order(order_id)
            final_status = await self.client.get_order_status(order_id)
            return final_status
        except:
            return {'status': 'timeout', 'filled_size': 0}
    
    def _calculate_sweep_price(
        self,
        orderbook: Dict[str, List],
        side: str,
        size: float
    ) -> float:
        """Calculate price to sweep the orderbook."""
        try:
            if side == 'buy':
                levels = orderbook.get('asks', [])
            else:
                levels = orderbook.get('bids', [])
            
            if not levels:
                raise ValueError("Empty orderbook")
            
            cumulative_size = 0
            sweep_price = 0
            
            for price, level_size in levels:
                cumulative_size += level_size
                sweep_price = price
                
                if cumulative_size >= size:
                    break
            
            return sweep_price
            
        except Exception as e:
            logger.error(f"Error calculating sweep price: {e}")
            # Return a safe default
            if side == 'buy':
                return levels[0][0] * 1.05 if levels else 0
            else:
                return levels[0][0] * 0.95 if levels else 0
    
    async def submit_limit_order_with_retry(
        self,
        order: Order,
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """Submit limit order with post-only retry logic."""
        attempts = 0
        
        while attempts < max_retries and order.post_only:
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
            
            if response.get('status') == 'success' or not response.get('post_only_reject'):
                return response
            
            # Adjust price for post-only
            attempts += 1
            if order.side == 'buy':
                order.price *= 0.9999  # Slightly lower bid
            else:
                order.price *= 1.0001  # Slightly higher ask
            
            # Validate adjusted price
            if order.price <= 0 or not np.isfinite(order.price):
                logger.error(f"Invalid adjusted price: {order.price}")
                return {'status': 'error', 'error': 'Price adjustment failed'}
            
            await asyncio.sleep(0.1)
        
        # Final attempt without post-only if needed
        if order.post_only and attempts >= max_retries:
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
    
    async def replace_with_aggressive_order(
        self,
        order: Order,
        aggressive_threshold: float
    ) -> Dict[str, Any]:
        """Replace order with more aggressive pricing."""
        try:
            # Cancel existing order
            await self.client.cancel_order(order.order_id)
            
            # Calculate aggressive price
            ticker = await self.client.get_ticker(order.symbol)
            
            if order.side == 'buy':
                aggressive_price = ticker['ask'] * 1.001
            else:
                aggressive_price = ticker['bid'] * 0.999
            
            # Place new order
            return await self.client.place_order(
                symbol=order.symbol,
                side=order.side,
                size=order.remaining_size,
                order_type='limit',
                price=aggressive_price,
                time_in_force='IOC'
            )
            
        except Exception as e:
            logger.error(f"Error replacing with aggressive order: {e}")
            return {'status': 'error', 'error': str(e)}
