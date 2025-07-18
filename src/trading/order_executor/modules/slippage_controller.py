"""
Slippage Controller Module

Manages slippage protection and calculation for order execution.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from .order_types import Order

logger = logging.getLogger(__name__)


class SlippageController:
    """
    Controls and monitors slippage during order execution.
    
    Provides slippage protection mechanisms and calculates
    actual slippage for executed orders.
    """
    
    def __init__(self):
        """Initialize the slippage controller."""
        self.slippage_history = []
        self.max_history = 1000
    
    async def execute_with_slippage_protection(
        self,
        client: Any,
        order: Order,
        max_slippage: float
    ) -> Dict[str, Any]:
        """
        Execute order with slippage protection.
        
        Args:
            client: Exchange client
            order: Order to execute
            max_slippage: Maximum allowed slippage
            
        Returns:
            Execution result
        """
        try:
            # Get current market price
            ticker = await client.get_ticker(order.symbol)
            
            # Validate ticker data
            if not ticker or 'bid' not in ticker or 'ask' not in ticker:
                logger.error(f"Invalid ticker data for {order.symbol}")
                return {'status': 'error', 'error': 'Invalid market data'}
            
            # Calculate expected price and max acceptable price
            if order.side == 'buy':
                expected_price = ticker['ask']
                # Validate price
                if not isinstance(expected_price, (int, float)) or expected_price <= 0:
                    return {'status': 'error', 'error': 'Invalid ask price'}
                max_price = expected_price * (1 + max_slippage)
            else:
                expected_price = ticker['bid']
                # Validate price
                if not isinstance(expected_price, (int, float)) or expected_price <= 0:
                    return {'status': 'error', 'error': 'Invalid bid price'}
                max_price = expected_price * (1 - max_slippage)
            
            # Store expected price for slippage calculation
            order.metadata['expected_price'] = expected_price
            
            # Submit as aggressive limit order
            result = await client.place_order(
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                order_type='limit',
                price=max_price,
                time_in_force='IOC',  # Immediate or cancel
                reduce_only=order.reduce_only
            )
            
            # Calculate and record slippage if filled
            if result.get('status') == 'success' and result.get('filled_size', 0) > 0:
                actual_price = result.get('avg_price', expected_price)
                slippage = self.calculate_slippage(order, expected_price, actual_price)
                
                result['slippage'] = slippage
                result['expected_price'] = expected_price
                
                # Record slippage
                self._record_slippage(order.symbol, order.side, slippage)
                
                # Check if slippage exceeded limit
                if abs(slippage) > max_slippage:
                    logger.warning(
                        f"Slippage exceeded limit: {slippage:.4f} > {max_slippage:.4f}"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing with slippage protection: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def calculate_slippage(
        self,
        order: Order,
        expected_price: Optional[float],
        actual_price: Optional[float] = None
    ) -> float:
        """
        Calculate slippage from expected price.
        
        Args:
            order: Executed order
            expected_price: Expected execution price
            actual_price: Actual execution price (uses order.avg_fill_price if None)
            
        Returns:
            Slippage percentage (positive = unfavorable)
        """
        try:
            if actual_price is None:
                actual_price = order.avg_fill_price
            
            if not expected_price or actual_price == 0:
                return 0
            
            # Validate prices
            if expected_price <= 0 or not np.isfinite(expected_price):
                return 0
            if actual_price <= 0 or not np.isfinite(actual_price):
                return 0
            
            if order.side == 'buy':
                # For buys, positive slippage means paying more
                slippage = (actual_price - expected_price) / expected_price
            else:
                # For sells, positive slippage means receiving less
                slippage = (expected_price - actual_price) / expected_price
            
            return slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return 0
    
    def _record_slippage(
        self,
        symbol: str,
        side: str,
        slippage: float
    ) -> None:
        """Record slippage for analysis."""
        try:
            record = {
                'timestamp': time.time(),
                'symbol': symbol,
                'side': side,
                'slippage': slippage
            }
            
            self.slippage_history.append(record)
            
            # Limit history size
            if len(self.slippage_history) > self.max_history:
                self.slippage_history = self.slippage_history[-self.max_history:]
                
        except Exception as e:
            logger.error(f"Error recording slippage: {e}")
    
    def get_slippage_statistics(
        self,
        symbol: Optional[str] = None,
        time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get slippage statistics.
        
        Args:
            symbol: Optional symbol filter
            time_window: Optional time window in seconds
            
        Returns:
            Slippage statistics
        """
        try:
            # Filter records
            records = self.slippage_history
            
            if symbol:
                records = [r for r in records if r['symbol'] == symbol]
            
            if time_window:
                cutoff = time.time() - time_window
                records = [r for r in records if r['timestamp'] > cutoff]
            
            if not records:
                return {'message': 'No slippage data available'}
            
            # Calculate statistics
            slippages = [r['slippage'] for r in records]
            
            return {
                'count': len(slippages),
                'mean_slippage': np.mean(slippages),
                'median_slippage': np.median(slippages),
                'std_slippage': np.std(slippages),
                'min_slippage': min(slippages),
                'max_slippage': max(slippages),
                'positive_slippage_rate': sum(1 for s in slippages if s > 0) / len(slippages),
                'avg_positive_slippage': np.mean([s for s in slippages if s > 0]) if any(s > 0 for s in slippages) else 0,
                'avg_negative_slippage': np.mean([s for s in slippages if s < 0]) if any(s < 0 for s in slippages) else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating slippage statistics: {e}")
            return {'error': str(e)}
    
    def estimate_slippage(
        self,
        symbol: str,
        side: str,
        size: float,
        orderbook: Optional[Dict[str, List]] = None
    ) -> float:
        """
        Estimate expected slippage for an order.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            size: Order size
            orderbook: Current orderbook (optional)
            
        Returns:
            Estimated slippage
        """
        try:
            # Use historical average if no orderbook
            if orderbook is None:
                historical = self.get_slippage_statistics(symbol)
                if 'mean_slippage' in historical:
                    return historical['mean_slippage']
                return 0.001  # Default 0.1%
            
            # Calculate from orderbook depth
            if side == 'buy':
                levels = orderbook.get('asks', [])
            else:
                levels = orderbook.get('bids', [])
            
            if not levels:
                return 0.001  # Default if no orderbook
            
            # Calculate volume-weighted average price
            cumulative_size = 0
            cumulative_cost = 0
            reference_price = levels[0][0] if levels else 0
            
            for price, level_size in levels:
                if cumulative_size >= size:
                    break
                
                fill_size = min(level_size, size - cumulative_size)
                cumulative_size += fill_size
                cumulative_cost += fill_size * price
            
            if cumulative_size > 0 and reference_price > 0:
                vwap = cumulative_cost / cumulative_size
                if side == 'buy':
                    estimated_slippage = (vwap - reference_price) / reference_price
                else:
                    estimated_slippage = (reference_price - vwap) / reference_price
                
                return max(0, estimated_slippage)  # Can't have negative slippage estimate
            
            return 0.001  # Default
            
        except Exception as e:
            logger.error(f"Error estimating slippage: {e}")
            return 0.001
    
    def should_split_order(
        self,
        symbol: str,
        side: str,
        size: float,
        max_slippage: float,
        orderbook: Optional[Dict[str, List]] = None
    ) -> bool:
        """
        Determine if order should be split to reduce slippage.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            size: Order size
            max_slippage: Maximum acceptable slippage
            orderbook: Current orderbook
            
        Returns:
            True if order should be split
        """
        try:
            estimated = self.estimate_slippage(symbol, side, size, orderbook)
            
            # Split if estimated slippage exceeds 80% of max
            return estimated > max_slippage * 0.8
            
        except Exception as e:
            logger.error(f"Error checking if should split order: {e}")
            return False
    
    def calculate_optimal_slice_size(
        self,
        symbol: str,
        side: str,
        total_size: float,
        max_slippage: float,
        orderbook: Optional[Dict[str, List]] = None
    ) -> float:
        """
        Calculate optimal slice size for order splitting.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            total_size: Total order size
            max_slippage: Maximum acceptable slippage
            orderbook: Current orderbook
            
        Returns:
            Optimal slice size
        """
        try:
            # Binary search for optimal size
            min_size = total_size * 0.05  # At least 5% per slice
            max_size = total_size
            
            while max_size - min_size > total_size * 0.01:  # 1% tolerance
                test_size = (min_size + max_size) / 2
                estimated = self.estimate_slippage(symbol, side, test_size, orderbook)
                
                if estimated > max_slippage:
                    max_size = test_size
                else:
                    min_size = test_size
            
            # Round to reasonable precision
            optimal_size = min_size
            if symbol.endswith('BTC'):
                optimal_size = round(optimal_size, 3)
            elif symbol.endswith('ETH'):
                optimal_size = round(optimal_size, 2)
            else:
                optimal_size = round(optimal_size, 4)
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal slice size: {e}")
            return total_size / 5  # Default to 5 slices
