"""
Order management module for market making strategy.
Handles order placement, cancellation, and tracking.
"""

import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order lifecycle for market making strategy"""
    
    def __init__(self):
        self.active_quotes = {}
        self.order_history = []
        self.max_order_age = 300  # 5 minutes
        
    async def place_quotes(self, quotes: List[Any], executor) -> Dict:
        """Place market making orders"""
        results = {
            'placed_orders': [],
            'failed_orders': [],
            'total_quotes': len(quotes) * 2  # Bid and ask for each level
        }
        
        if not quotes:
            return results
        
        try:
            # Validate executor
            if not executor:
                raise ValueError("No executor available")
            
            tasks = []
            
            for quote in quotes:
                try:
                    # Cancel existing orders at this level
                    if quote.symbol in self.active_quotes:
                        await self._cancel_existing_orders(quote.symbol, executor)
                    
                    # Prepare bid and ask orders
                    bid_task = self._prepare_order_task(
                        executor, quote, 'bid',
                        quote.symbol, quote.bid_price, quote.bid_size
                    )
                    tasks.append(('bid', quote, bid_task))
                    
                    ask_task = self._prepare_order_task(
                        executor, quote, 'ask',
                        quote.symbol, quote.ask_price, quote.ask_size
                    )
                    tasks.append(('ask', quote, ask_task))
                    
                except Exception as e:
                    logger.error(f"Error preparing orders for {quote.symbol}: {e}")
                    results['failed_orders'].append({
                        'quote': quote,
                        'error': str(e)
                    })
            
            # Execute all orders
            if tasks:
                await self._execute_order_tasks(tasks, results)
            
            # Calculate success rate
            success_rate = len(results['placed_orders']) / results['total_quotes'] if results['total_quotes'] > 0 else 0
            results['success_rate'] = success_rate
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in place_quotes: {e}")
            return results
    
    async def _cancel_existing_orders(self, symbol: str, executor):
        """Cancel existing orders for a symbol"""
        if symbol not in self.active_quotes:
            return
        
        try:
            cancel_tasks = []
            current_time = pd.Timestamp.now()
            active_orders = []
            
            # Filter out stale orders
            for quote in self.active_quotes[symbol]:
                if 'timestamp' in quote:
                    age = (current_time - quote['timestamp']).seconds
                    if age < self.max_order_age:
                        active_orders.append(quote)
                        cancel_tasks.append(
                            executor.cancel_order_async(symbol, quote['order_id'])
                        )
                else:
                    active_orders.append(quote)
                    cancel_tasks.append(
                        executor.cancel_order_async(symbol, quote['order_id'])
                    )
            
            if cancel_tasks:
                # Cancel with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*cancel_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                    
                    success_count = sum(1 for r in results if not isinstance(r, Exception))
                    logger.debug(f"Cancelled {success_count}/{len(cancel_tasks)} orders for {symbol}")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout cancelling orders for {symbol}")
            
            # Clear active quotes
            self.active_quotes[symbol] = []
            
        except Exception as e:
            logger.error(f"Error cancelling quotes for {symbol}: {e}")
            # Clear anyway to avoid stale state
            self.active_quotes[symbol] = []
    
    def _prepare_order_task(self, executor, quote: Any, order_type: str,
                           symbol: str, price: float, size: float):
        """Prepare an order placement task"""
        return executor.place_order_async(
            symbol=symbol,
            side='buy' if order_type == 'bid' else 'sell',
            size=size,
            price=price,
            type='limit',
            time_in_force='GTC',
            post_only=True  # Ensure maker fees
        )
    
    async def _execute_order_tasks(self, tasks: List, results: Dict):
        """Execute order placement tasks with error handling"""
        timeout = 10.0  # 10 second timeout for all orders
        
        try:
            task_results = await asyncio.wait_for(
                asyncio.gather(*[t[2] for t in tasks], return_exceptions=True),
                timeout=timeout
            )
            
            # Process results
            for i, (order_type, quote, _) in enumerate(tasks):
                result = task_results[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Order placement failed: {result}")
                    results['failed_orders'].append({
                        'type': order_type,
                        'quote': quote,
                        'error': str(result)
                    })
                elif isinstance(result, dict) and result.get('status') == 'accepted':
                    results['placed_orders'].append({
                        'type': order_type,
                        'quote': quote,
                        'order': result
                    })
                    
                    # Track active order
                    self._track_active_order(quote.symbol, result, order_type, quote)
                else:
                    results['failed_orders'].append({
                        'type': order_type,
                        'quote': quote,
                        'reason': result.get('reason', 'unknown') if isinstance(result, dict) else 'unknown'
                    })
                    
        except asyncio.TimeoutError:
            logger.error("Timeout placing orders")
            results['failed_orders'].extend([{
                'type': t[0],
                'quote': t[1],
                'error': 'timeout'
            } for t in tasks])
        except Exception as e:
            logger.error(f"Error executing orders: {e}")
    
    def _track_active_order(self, symbol: str, order_result: Dict, 
                           order_type: str, quote: Any):
        """Track an active order"""
        if symbol not in self.active_quotes:
            self.active_quotes[symbol] = []
        
        self.active_quotes[symbol].append({
            'order_id': order_result['order_id'],
            'side': order_type,
            'price': quote.bid_price if order_type == 'bid' else quote.ask_price,
            'size': quote.bid_size if order_type == 'bid' else quote.ask_size,
            'timestamp': pd.Timestamp.now()
        })
        
        # Add to history
        self.order_history.append({
            'symbol': symbol,
            'order_id': order_result['order_id'],
            'side': order_type,
            'price': quote.bid_price if order_type == 'bid' else quote.ask_price,
            'size': quote.bid_size if order_type == 'bid' else quote.ask_size,
            'timestamp': pd.Timestamp.now(),
            'status': 'active'
        })
    
    async def cancel_all_orders(self, executor, symbols: Optional[List[str]] = None):
        """Cancel all active orders"""
        try:
            if symbols is None:
                symbols = list(self.active_quotes.keys())
            
            cancel_tasks = []
            
            for symbol in symbols:
                if symbol in self.active_quotes:
                    for order in self.active_quotes[symbol]:
                        cancel_tasks.append(
                            executor.cancel_order_async(symbol, order['order_id'])
                        )
            
            if cancel_tasks:
                results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                logger.info(f"Cancelled {success_count}/{len(cancel_tasks)} orders")
                
                # Clear active quotes
                for symbol in symbols:
                    self.active_quotes.pop(symbol, None)
                    
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
    
    def get_active_orders_count(self) -> Dict[str, int]:
        """Get count of active orders by symbol"""
        return {
            symbol: len(orders) 
            for symbol, orders in self.active_quotes.items()
        }
    
    def get_order_metrics(self) -> Dict:
        """Get order management metrics"""
        try:
            total_active = sum(len(orders) for orders in self.active_quotes.values())
            
            # Calculate order age distribution
            age_distribution = {'0-60s': 0, '60-180s': 0, '180-300s': 0, '>300s': 0}
            current_time = pd.Timestamp.now()
            
            for orders in self.active_quotes.values():
                for order in orders:
                    if 'timestamp' in order:
                        age = (current_time - order['timestamp']).seconds
                        if age < 60:
                            age_distribution['0-60s'] += 1
                        elif age < 180:
                            age_distribution['60-180s'] += 1
                        elif age < 300:
                            age_distribution['180-300s'] += 1
                        else:
                            age_distribution['>300s'] += 1
            
            return {
                'total_active_orders': total_active,
                'active_symbols': len(self.active_quotes),
                'age_distribution': age_distribution,
                'total_historical_orders': len(self.order_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting order metrics: {e}")
            return {}
    
    def cleanup_stale_orders(self):
        """Remove stale orders from tracking"""
        try:
            current_time = pd.Timestamp.now()
            
            for symbol in list(self.active_quotes.keys()):
                active_orders = []
                
                for order in self.active_quotes[symbol]:
                    if 'timestamp' in order:
                        age = (current_time - order['timestamp']).seconds
                        if age < self.max_order_age * 2:  # Keep for 2x max age
                            active_orders.append(order)
                    else:
                        active_orders.append(order)
                
                if active_orders:
                    self.active_quotes[symbol] = active_orders
                else:
                    del self.active_quotes[symbol]
                    
        except Exception as e:
            logger.error(f"Error cleaning up stale orders: {e}")