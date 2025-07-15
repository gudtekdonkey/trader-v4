"""
Advanced Order Execution Algorithms
TWAP, VWAP, and other sophisticated execution strategies
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class AdvancedOrderExecutor:
    """Advanced order execution with TWAP, VWAP, and other algorithms"""
    
    def __init__(self, exchange_client):
        self.exchange = exchange_client
        self.active_orders = {}
        self.execution_history = []
        
    async def execute_twap_order(self, 
                               symbol: str,
                               side: str,
                               total_size: float,
                               duration_minutes: int,
                               metadata: Optional[Dict] = None) -> Dict:
        """
        Execute TWAP (Time-Weighted Average Price) order
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_size: Total size to execute
            duration_minutes: Duration over which to execute
            metadata: Additional order metadata
            
        Returns:
            Execution result dictionary
        """
        try:
            logger.info(f"Starting TWAP execution: {symbol} {side} {total_size} over {duration_minutes}m")
            
            # Calculate number of child orders
            interval_minutes = max(1, duration_minutes // 10)  # At least 10 slices
            n_slices = duration_minutes // interval_minutes
            slice_size = total_size / n_slices
            
            execution_id = f"twap_{symbol}_{int(time.time())}"
            
            execution_result = {
                'execution_id': execution_id,
                'symbol': symbol,
                'side': side,
                'total_size': total_size,
                'duration_minutes': duration_minutes,
                'start_time': datetime.now(),
                'child_orders': [],
                'total_filled': 0,
                'average_price': 0,
                'status': 'running',
                'metadata': metadata or {}
            }
            
            self.active_orders[execution_id] = execution_result
            
            # Execute child orders
            for i in range(n_slices):
                try:
                    # Wait for next execution time
                    if i > 0:
                        await asyncio.sleep(interval_minutes * 60)
                    
                    # Check if we should continue
                    if execution_id not in self.active_orders:
                        break
                    
                    # Adjust slice size for last order
                    remaining_size = total_size - execution_result['total_filled']
                    current_slice_size = min(slice_size, remaining_size)
                    
                    if current_slice_size <= 0:
                        break
                    
                    # Place child order
                    child_order = await self._place_child_order(
                        symbol=symbol,
                        side=side,
                        size=current_slice_size,
                        slice_number=i+1,
                        execution_id=execution_id
                    )
                    
                    if child_order:
                        execution_result['child_orders'].append(child_order)
                        
                        if child_order.get('status') in ['filled', 'partial']:
                            filled_size = child_order.get('filled_size', 0)
                            fill_price = child_order.get('fill_price', 0)
                            
                            # Update average price
                            old_total = execution_result['total_filled']
                            old_avg = execution_result['average_price']
                            
                            execution_result['total_filled'] += filled_size
                            
                            if execution_result['total_filled'] > 0:
                                execution_result['average_price'] = (
                                    (old_total * old_avg + filled_size * fill_price) / 
                                    execution_result['total_filled']
                                )
                    
                    logger.info(f"TWAP slice {i+1}/{n_slices} executed: {current_slice_size:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in TWAP slice {i+1}: {e}")
                    continue
            
            # Finalize execution
            execution_result['end_time'] = datetime.now()
            execution_result['status'] = 'completed'
            
            fill_rate = execution_result['total_filled'] / total_size if total_size > 0 else 0
            execution_result['fill_rate'] = fill_rate
            
            logger.info(f"TWAP execution completed: {fill_rate:.2%} filled at avg price {execution_result['average_price']:.4f}")
            
            # Store in history
            self.execution_history.append(execution_result.copy())
            
            # Clean up active order
            if execution_id in self.active_orders:
                del self.active_orders[execution_id]
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error in TWAP execution: {e}")
            return {
                'error': str(e),
                'status': 'failed',
                'symbol': symbol,
                'side': side,
                'total_size': total_size
            }
    
    async def execute_vwap_order(self,
                               symbol: str,
                               side: str,
                               total_size: float,
                               duration_minutes: int,
                               historical_volume_window: int = 20) -> Dict:
        """
        Execute VWAP (Volume-Weighted Average Price) order
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_size: Total size to execute
            duration_minutes: Duration over which to execute
            historical_volume_window: Days of historical volume for weighting
            
        Returns:
            Execution result dictionary
        """
        try:
            logger.info(f"Starting VWAP execution: {symbol} {side} {total_size}")
            
            # Get historical volume profile
            volume_profile = await self._get_volume_profile(symbol, historical_volume_window)
            
            if not volume_profile:
                # Fallback to TWAP if no volume data
                logger.warning("No volume data available, falling back to TWAP")
                return await self.execute_twap_order(symbol, side, total_size, duration_minutes)
            
            execution_id = f"vwap_{symbol}_{int(time.time())}"
            
            execution_result = {
                'execution_id': execution_id,
                'symbol': symbol,
                'side': side,
                'total_size': total_size,
                'start_time': datetime.now(),
                'child_orders': [],
                'total_filled': 0,
                'average_price': 0,
                'status': 'running',
                'volume_profile': volume_profile
            }
            
            self.active_orders[execution_id] = execution_result
            
            # Calculate VWAP schedule
            vwap_schedule = self._calculate_vwap_schedule(
                total_size, duration_minutes, volume_profile
            )
            
            # Execute according to VWAP schedule
            for i, schedule_item in enumerate(vwap_schedule):
                try:
                    # Wait for scheduled time
                    if i > 0:
                        await asyncio.sleep(schedule_item['wait_minutes'] * 60)
                    
                    # Place order with volume-weighted size
                    child_order = await self._place_child_order(
                        symbol=symbol,
                        side=side,
                        size=schedule_item['size'],
                        slice_number=i+1,
                        execution_id=execution_id
                    )
                    
                    if child_order:
                        execution_result['child_orders'].append(child_order)
                        
                        if child_order.get('status') in ['filled', 'partial']:
                            filled_size = child_order.get('filled_size', 0)
                            fill_price = child_order.get('fill_price', 0)
                            
                            # Update totals
                            old_total = execution_result['total_filled']
                            old_avg = execution_result['average_price']
                            
                            execution_result['total_filled'] += filled_size
                            
                            if execution_result['total_filled'] > 0:
                                execution_result['average_price'] = (
                                    (old_total * old_avg + filled_size * fill_price) / 
                                    execution_result['total_filled']
                                )
                    
                except Exception as e:
                    logger.error(f"Error in VWAP slice {i+1}: {e}")
                    continue
            
            # Finalize
            execution_result['end_time'] = datetime.now()
            execution_result['status'] = 'completed'
            execution_result['fill_rate'] = execution_result['total_filled'] / total_size
            
            self.execution_history.append(execution_result.copy())
            
            if execution_id in self.active_orders:
                del self.active_orders[execution_id]
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error in VWAP execution: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def execute_iceberg_order(self,
                                  symbol: str,
                                  side: str,
                                  total_size: float,
                                  visible_size: float,
                                  price: Optional[float] = None) -> Dict:
        """
        Execute iceberg order (large order hidden by showing only small portions)
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_size: Total order size
            visible_size: Size visible in order book at any time
            price: Limit price (None for market price)
            
        Returns:
            Execution result dictionary
        """
        try:
            logger.info(f"Starting iceberg execution: {symbol} {side} {total_size} (visible: {visible_size})")
            
            execution_id = f"iceberg_{symbol}_{int(time.time())}"
            
            execution_result = {
                'execution_id': execution_id,
                'symbol': symbol,
                'side': side,
                'total_size': total_size,
                'visible_size': visible_size,
                'start_time': datetime.now(),
                'child_orders': [],
                'total_filled': 0,
                'average_price': 0,
                'status': 'running'
            }
            
            self.active_orders[execution_id] = execution_result
            
            remaining_size = total_size
            
            while remaining_size > 0 and execution_id in self.active_orders:
                # Calculate current slice size
                current_slice = min(visible_size, remaining_size)
                
                # Place visible portion
                child_order = await self._place_child_order(
                    symbol=symbol,
                    side=side,
                    size=current_slice,
                    price=price,
                    execution_id=execution_id
                )
                
                if child_order:
                    execution_result['child_orders'].append(child_order)
                    
                    if child_order.get('status') in ['filled', 'partial']:
                        filled_size = child_order.get('filled_size', 0)
                        fill_price = child_order.get('fill_price', 0)
                        
                        # Update totals
                        old_total = execution_result['total_filled']
                        old_avg = execution_result['average_price']
                        
                        execution_result['total_filled'] += filled_size
                        remaining_size -= filled_size
                        
                        if execution_result['total_filled'] > 0:
                            execution_result['average_price'] = (
                                (old_total * old_avg + filled_size * fill_price) / 
                                execution_result['total_filled']
                            )
                
                # Wait a bit before next slice
                await asyncio.sleep(5)  # 5 second delay
            
            execution_result['end_time'] = datetime.now()
            execution_result['status'] = 'completed'
            execution_result['fill_rate'] = execution_result['total_filled'] / total_size
            
            self.execution_history.append(execution_result.copy())
            
            if execution_id in self.active_orders:
                del self.active_orders[execution_id]
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error in iceberg execution: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _place_child_order(self,
                               symbol: str,
                               side: str,
                               size: float,
                               slice_number: int = 1,
                               execution_id: str = None,
                               price: Optional[float] = None) -> Optional[Dict]:
        """
        Place individual child order as part of larger execution
        """
        try:
            # Get current market price if no price specified
            if price is None:
                ticker = await self.exchange.get_ticker(symbol)
                if side == 'buy':
                    price = ticker.get('ask', ticker.get('last', 0))
                else:
                    price = ticker.get('bid', ticker.get('last', 0))
            
            # Place order through exchange
            order_result = await self.exchange.place_order(
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
            
            # Wait for fill (simplified)
            await asyncio.sleep(1)
            
            # Mock fill for demo (replace with actual exchange integration)
            fill_result = {
                'order_id': order_result.get('id', f"child_{slice_number}"),
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'fill_price': price * (1 + np.random.normal(0, 0.001)),  # Small slippage
                'filled_size': size * np.random.uniform(0.95, 1.0),  # Partial fill simulation
                'status': 'filled',
                'timestamp': datetime.now(),
                'execution_id': execution_id,
                'slice_number': slice_number
            }
            
            logger.debug(f"Child order executed: {fill_result}")
            return fill_result
            
        except Exception as e:
            logger.error(f"Error placing child order: {e}")
            return None
    
    async def _get_volume_profile(self, symbol: str, days: int = 20) -> Optional[List[Dict]]:
        """
        Get historical volume profile for VWAP calculation
        """
        try:
            # This would typically fetch real volume data
            # For demo, return mock intraday volume profile
            
            # Typical crypto volume profile (higher during certain hours)
            hours = list(range(24))
            
            # Mock volume weights (higher volume during active trading hours)
            volume_weights = [
                0.02, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5
                0.04, 0.06, 0.08, 0.09, 0.08, 0.07,  # 6-11
                0.06, 0.08, 0.09, 0.08, 0.07, 0.06,  # 12-17
                0.05, 0.04, 0.03, 0.03, 0.02, 0.02   # 18-23
            ]
            
            profile = []
            for hour, weight in zip(hours, volume_weights):
                profile.append({
                    'hour': hour,
                    'volume_weight': weight,
                    'avg_volume': weight * 1000000  # Mock volume
                })
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting volume profile: {e}")
            return None
    
    def _calculate_vwap_schedule(self, 
                               total_size: float, 
                               duration_minutes: int, 
                               volume_profile: List[Dict]) -> List[Dict]:
        """
        Calculate VWAP execution schedule based on volume profile
        """
        try:
            current_hour = datetime.now().hour
            
            # Find relevant volume weights for execution period
            execution_hours = duration_minutes // 60 + 1
            relevant_weights = []
            
            for i in range(execution_hours):
                hour = (current_hour + i) % 24
                hour_profile = next((p for p in volume_profile if p['hour'] == hour), None)
                if hour_profile:
                    relevant_weights.append(hour_profile['volume_weight'])
                else:
                    relevant_weights.append(1.0 / 24)  # Equal weight fallback
            
            # Normalize weights
            total_weight = sum(relevant_weights)
            normalized_weights = [w / total_weight for w in relevant_weights]
            
            # Create schedule
            schedule = []
            interval_minutes = duration_minutes // len(normalized_weights)
            
            for i, weight in enumerate(normalized_weights):
                schedule.append({
                    'slice_number': i + 1,
                    'size': total_size * weight,
                    'wait_minutes': interval_minutes if i > 0 else 0,
                    'volume_weight': weight
                })
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error calculating VWAP schedule: {e}")
            # Fallback to equal distribution
            n_slices = max(1, duration_minutes // 10)
            slice_size = total_size / n_slices
            return [{'slice_number': i+1, 'size': slice_size, 'wait_minutes': 10 if i > 0 else 0} 
                   for i in range(n_slices)]
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel active execution algorithm
        """
        try:
            if execution_id in self.active_orders:
                execution = self.active_orders[execution_id]
                execution['status'] = 'cancelled'
                execution['end_time'] = datetime.now()
                
                # Move to history
                self.execution_history.append(execution.copy())
                del self.active_orders[execution_id]
                
                logger.info(f"Execution {execution_id} cancelled")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling execution: {e}")
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """
        Get status of active or completed execution
        """
        if execution_id in self.active_orders:
            return self.active_orders[execution_id].copy()
        
        # Check history
        for execution in self.execution_history:
            if execution.get('execution_id') == execution_id:
                return execution.copy()
        
        return None
    
    def get_execution_statistics(self) -> Dict:
        """
        Get execution performance statistics
        """
        try:
            if not self.execution_history:
                return {}
            
            completed_executions = [e for e in self.execution_history if e.get('status') == 'completed']
            
            if not completed_executions:
                return {}
            
            # Calculate statistics
            fill_rates = [e.get('fill_rate', 0) for e in completed_executions]
            execution_times = []
            
            for e in completed_executions:
                if 'start_time' in e and 'end_time' in e:
                    duration = (e['end_time'] - e['start_time']).total_seconds() / 60
                    execution_times.append(duration)
            
            stats = {
                'total_executions': len(completed_executions),
                'average_fill_rate': np.mean(fill_rates) if fill_rates else 0,
                'min_fill_rate': np.min(fill_rates) if fill_rates else 0,
                'max_fill_rate': np.max(fill_rates) if fill_rates else 0,
                'average_execution_time_minutes': np.mean(execution_times) if execution_times else 0,
                'successful_executions': len([e for e in completed_executions if e.get('fill_rate', 0) > 0.95]),
                'active_executions': len(self.active_orders)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating execution statistics: {e}")
            return {}

# Usage example
if __name__ == '__main__':
    import asyncio
    
    async def test_twap():
        # Mock exchange client
        class MockExchange:
            async def get_ticker(self, symbol):
                return {'bid': 50000, 'ask': 50010, 'last': 50005}
            
            async def place_order(self, **kwargs):
                return {'id': f"order_{int(time.time())}", 'status': 'placed'}
        
        # Test TWAP execution
        executor = AdvancedOrderExecutor(MockExchange())
        
        result = await executor.execute_twap_order(
            symbol="BTC-USD",
            side="buy",
            total_size=1.0,
            duration_minutes=30
        )
        
        print(f"TWAP Result: {result}")
        print(f"Statistics: {executor.get_execution_statistics()}")
    
    # Run test
    asyncio.run(test_twap())