"""
Execution algorithms module for advanced order execution.
Contains TWAP, VWAP, and Iceberg order execution logic.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class TWAPExecutor:
    """Time-Weighted Average Price execution algorithm"""
    
    def __init__(self, executor):
        self.executor = executor
        
    async def execute(self, 
                     symbol: str,
                     side: str,
                     total_size: float,
                     duration_minutes: int,
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Execute TWAP order by splitting into time-based slices
        
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
            # Validate inputs
            self._validate_inputs(symbol, side, total_size, duration_minutes)
            
            # Check execution limits
            if len(self.executor.active_orders) >= self.executor.max_active_executions:
                raise RuntimeError(f"Maximum active executions ({self.executor.max_active_executions}) reached")
            
            logger.info(f"Starting TWAP execution: {symbol} {side} {total_size} over {duration_minutes}m")
            
            # Calculate slicing parameters
            slicing_params = self._calculate_slicing_params(total_size, duration_minutes)
            
            execution_id = f"twap_{symbol}_{int(time.time())}"
            
            execution_result = self._initialize_execution_result(
                execution_id, symbol, side, total_size, duration_minutes, metadata
            )
            
            self.executor.active_orders[execution_id] = execution_result
            self.executor.execution_metrics['total_executions'] += 1
            
            # Execute slices
            await self._execute_slices(execution_result, slicing_params)
            
            # Finalize execution
            return self._finalize_execution(execution_result, total_size)
            
        except Exception as e:
            logger.error(f"Error in TWAP execution: {e}")
            self.executor.execution_metrics['failed_executions'] += 1
            
            return {
                'error': str(e),
                'status': 'failed',
                'symbol': symbol,
                'side': side,
                'total_size': total_size
            }
    
    def _validate_inputs(self, symbol: str, side: str, total_size: float, duration_minutes: int):
        """Validate TWAP order inputs"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid symbol")
            
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}")
            
        if not isinstance(total_size, (int, float)) or total_size <= 0:
            raise ValueError(f"Invalid total size: {total_size}")
            
        if total_size < self.executor.min_order_size:
            raise ValueError(f"Order size {total_size} below minimum {self.executor.min_order_size}")
            
        if not isinstance(duration_minutes, int) or duration_minutes <= 0:
            raise ValueError(f"Invalid duration: {duration_minutes}")
            
        if duration_minutes > self.executor.max_execution_time_hours * 60:
            raise ValueError(f"Duration exceeds maximum of {self.executor.max_execution_time_hours} hours")
    
    def _calculate_slicing_params(self, total_size: float, duration_minutes: int) -> Dict:
        """Calculate TWAP slicing parameters"""
        interval_minutes = max(1, duration_minutes // 10)  # At least 10 slices
        n_slices = max(2, duration_minutes // interval_minutes)  # At least 2 slices
        slice_size = total_size / n_slices
        
        # Ensure slice size is above minimum
        if slice_size < self.executor.min_order_size:
            n_slices = int(total_size / self.executor.min_order_size)
            slice_size = total_size / n_slices
            interval_minutes = duration_minutes // n_slices
        
        return {
            'n_slices': n_slices,
            'slice_size': slice_size,
            'interval_minutes': interval_minutes
        }
    
    def _initialize_execution_result(self, execution_id: str, symbol: str, 
                                   side: str, total_size: float, 
                                   duration_minutes: int, metadata: Optional[Dict]) -> Dict:
        """Initialize execution result dictionary"""
        return {
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
            'metadata': metadata or {},
            'errors': []
        }
    
    async def _execute_slices(self, execution_result: Dict, slicing_params: Dict):
        """Execute TWAP slices"""
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        for i in range(slicing_params['n_slices']):
            try:
                # Check if execution should continue
                if execution_result['execution_id'] not in self.executor.active_orders:
                    logger.warning(f"Execution {execution_result['execution_id']} cancelled")
                    break
                    
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures in TWAP execution")
                    execution_result['status'] = 'failed'
                    break
                
                # Wait for next execution time
                if i > 0:
                    await asyncio.sleep(slicing_params['interval_minutes'] * 60)
                
                # Calculate current slice size
                remaining_size = execution_result['total_size'] - execution_result['total_filled']
                current_slice_size = min(slicing_params['slice_size'], remaining_size)
                
                if current_slice_size < self.executor.min_order_size:
                    logger.info(f"Remaining size {current_slice_size} below minimum, stopping")
                    break
                
                # Place child order with retries
                child_order = await self.executor._place_child_order_with_retries(
                    symbol=execution_result['symbol'],
                    side=execution_result['side'],
                    size=current_slice_size,
                    slice_number=i+1,
                    execution_id=execution_result['execution_id']
                )
                
                # Process child order result
                if child_order and child_order.get('status') != 'error':
                    execution_result['child_orders'].append(child_order)
                    consecutive_failures = 0
                    
                    self._update_execution_totals(execution_result, child_order)
                else:
                    consecutive_failures += 1
                    execution_result['errors'].append(f"Failed slice {i+1}")
                
                logger.info(f"TWAP slice {i+1}/{slicing_params['n_slices']} executed: {current_slice_size:.4f}")
                
            except asyncio.CancelledError:
                logger.info(f"TWAP execution {execution_result['execution_id']} cancelled")
                execution_result['status'] = 'cancelled'
                break
            except Exception as e:
                logger.error(f"Error in TWAP slice {i+1}: {e}")
                execution_result['errors'].append(f"Slice {i+1} error: {str(e)}")
                consecutive_failures += 1
                continue
    
    def _update_execution_totals(self, execution_result: Dict, child_order: Dict):
        """Update execution totals with child order results"""
        if child_order.get('status') in ['filled', 'partial']:
            filled_size = child_order.get('filled_size', 0)
            fill_price = child_order.get('fill_price', 0)
            
            if filled_size > 0 and fill_price > 0:
                # Update average price
                old_total = execution_result['total_filled']
                old_avg = execution_result['average_price']
                
                execution_result['total_filled'] += filled_size
                
                if execution_result['total_filled'] > 0:
                    execution_result['average_price'] = (
                        (old_total * old_avg + filled_size * fill_price) / 
                        execution_result['total_filled']
                    )
    
    def _finalize_execution(self, execution_result: Dict, total_size: float) -> Dict:
        """Finalize execution and update metrics"""
        execution_result['end_time'] = datetime.now()
        
        if execution_result['status'] == 'running':
            execution_result['status'] = 'completed'
            self.executor.execution_metrics['successful_executions'] += 1
        elif execution_result['status'] == 'failed':
            self.executor.execution_metrics['failed_executions'] += 1
        
        fill_rate = execution_result['total_filled'] / total_size if total_size > 0 else 0
        execution_result['fill_rate'] = fill_rate
        
        if 0 < fill_rate < 0.95:
            self.executor.execution_metrics['partial_executions'] += 1
        
        logger.info(f"TWAP execution completed: {fill_rate:.2%} filled at avg price {execution_result['average_price']:.4f}")
        
        # Store in history
        self.executor._add_to_history(execution_result)
        
        # Clean up active order
        if execution_result['execution_id'] in self.executor.active_orders:
            del self.executor.active_orders[execution_result['execution_id']]
        
        return execution_result


class VWAPExecutor:
    """Volume-Weighted Average Price execution algorithm"""
    
    def __init__(self, executor):
        self.executor = executor
        
    async def execute(self,
                     symbol: str,
                     side: str,
                     total_size: float,
                     duration_minutes: int,
                     historical_volume_window: int = 20) -> Dict:
        """
        Execute VWAP order using historical volume profile
        
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
            # Validate inputs
            self._validate_inputs(symbol, side, total_size, duration_minutes)
            
            logger.info(f"Starting VWAP execution: {symbol} {side} {total_size}")
            
            # Get historical volume profile
            volume_profile = await self.executor._get_volume_profile_safe(symbol, historical_volume_window)
            
            if not volume_profile:
                # Fallback to TWAP if no volume data
                logger.warning("No volume data available, falling back to TWAP")
                twap_executor = TWAPExecutor(self.executor)
                return await twap_executor.execute(symbol, side, total_size, duration_minutes)
            
            execution_id = f"vwap_{symbol}_{int(time.time())}"
            
            execution_result = self._initialize_execution_result(
                execution_id, symbol, side, total_size, volume_profile
            )
            
            self.executor.active_orders[execution_id] = execution_result
            self.executor.execution_metrics['total_executions'] += 1
            
            # Calculate VWAP schedule
            vwap_schedule = self.executor._calculate_vwap_schedule(
                total_size, duration_minutes, volume_profile
            )
            
            # Execute according to VWAP schedule
            await self._execute_vwap_schedule(execution_result, vwap_schedule)
            
            # Finalize
            return self._finalize_execution(execution_result, total_size)
            
        except Exception as e:
            logger.error(f"Error in VWAP execution: {e}")
            self.executor.execution_metrics['failed_executions'] += 1
            return {'error': str(e), 'status': 'failed'}
    
    def _validate_inputs(self, symbol: str, side: str, total_size: float, duration_minutes: int):
        """Validate VWAP order inputs"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid symbol")
            
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}")
            
        if not isinstance(total_size, (int, float)) or total_size <= 0:
            raise ValueError(f"Invalid total size: {total_size}")
            
        if not isinstance(duration_minutes, int) or duration_minutes <= 0:
            raise ValueError(f"Invalid duration: {duration_minutes}")
    
    def _initialize_execution_result(self, execution_id: str, symbol: str,
                                   side: str, total_size: float,
                                   volume_profile: List[Dict]) -> Dict:
        """Initialize VWAP execution result"""
        return {
            'execution_id': execution_id,
            'symbol': symbol,
            'side': side,
            'total_size': total_size,
            'start_time': datetime.now(),
            'child_orders': [],
            'total_filled': 0,
            'average_price': 0,
            'status': 'running',
            'volume_profile': volume_profile,
            'errors': []
        }
    
    async def _execute_vwap_schedule(self, execution_result: Dict, vwap_schedule: List[Dict]):
        """Execute orders according to VWAP schedule"""
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        for i, schedule_item in enumerate(vwap_schedule):
            try:
                # Check execution health
                if execution_result['execution_id'] not in self.executor.active_orders:
                    break
                    
                if consecutive_failures >= max_consecutive_failures:
                    execution_result['status'] = 'failed'
                    break
                
                # Wait for scheduled time
                if i > 0 and schedule_item.get('wait_minutes', 0) > 0:
                    await asyncio.sleep(schedule_item['wait_minutes'] * 60)
                
                # Validate schedule item
                scheduled_size = schedule_item.get('size', 0)
                if scheduled_size < self.executor.min_order_size:
                    logger.debug(f"Skipping slice {i+1} with size {scheduled_size}")
                    continue
                
                # Place order with volume-weighted size
                child_order = await self.executor._place_child_order_with_retries(
                    symbol=execution_result['symbol'],
                    side=execution_result['side'],
                    size=scheduled_size,
                    slice_number=i+1,
                    execution_id=execution_result['execution_id']
                )
                
                # Process result
                if child_order and child_order.get('status') != 'error':
                    execution_result['child_orders'].append(child_order)
                    consecutive_failures = 0
                    
                    self._update_execution_totals(execution_result, child_order)
                else:
                    consecutive_failures += 1
                    execution_result['errors'].append(f"Failed VWAP slice {i+1}")
                
            except Exception as e:
                logger.error(f"Error in VWAP slice {i+1}: {e}")
                execution_result['errors'].append(f"VWAP slice {i+1} error: {str(e)}")
                consecutive_failures += 1
                continue
    
    def _update_execution_totals(self, execution_result: Dict, child_order: Dict):
        """Update execution totals with child order results"""
        if child_order.get('status') in ['filled', 'partial']:
            filled_size = child_order.get('filled_size', 0)
            fill_price = child_order.get('fill_price', 0)
            
            if filled_size > 0 and fill_price > 0:
                # Update totals
                old_total = execution_result['total_filled']
                old_avg = execution_result['average_price']
                
                execution_result['total_filled'] += filled_size
                
                if execution_result['total_filled'] > 0:
                    execution_result['average_price'] = (
                        (old_total * old_avg + filled_size * fill_price) / 
                        execution_result['total_filled']
                    )
    
    def _finalize_execution(self, execution_result: Dict, total_size: float) -> Dict:
        """Finalize VWAP execution"""
        execution_result['end_time'] = datetime.now()
        
        if execution_result['status'] == 'running':
            execution_result['status'] = 'completed'
            self.executor.execution_metrics['successful_executions'] += 1
        else:
            self.executor.execution_metrics['failed_executions'] += 1
            
        execution_result['fill_rate'] = execution_result['total_filled'] / total_size
        
        self.executor._add_to_history(execution_result)
        
        if execution_result['execution_id'] in self.executor.active_orders:
            del self.executor.active_orders[execution_result['execution_id']]
        
        return execution_result


class IcebergExecutor:
    """Iceberg order execution algorithm"""
    
    def __init__(self, executor):
        self.executor = executor
        
    async def execute(self,
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
            # Validate inputs
            self._validate_inputs(symbol, side, total_size, visible_size, price)
            
            logger.info(f"Starting iceberg execution: {symbol} {side} {total_size} (visible: {visible_size})")
            
            execution_id = f"iceberg_{symbol}_{int(time.time())}"
            
            execution_result = self._initialize_execution_result(
                execution_id, symbol, side, total_size, visible_size
            )
            
            self.executor.active_orders[execution_id] = execution_result
            self.executor.execution_metrics['total_executions'] += 1
            
            # Execute iceberg slices
            await self._execute_iceberg_slices(execution_result, total_size, visible_size, price)
            
            # Finalize
            return self._finalize_execution(execution_result, total_size)
            
        except Exception as e:
            logger.error(f"Error in iceberg execution: {e}")
            self.executor.execution_metrics['failed_executions'] += 1
            return {'error': str(e), 'status': 'failed'}
    
    def _validate_inputs(self, symbol: str, side: str, total_size: float, 
                        visible_size: float, price: Optional[float]):
        """Validate iceberg order inputs"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid symbol")
            
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}")
            
        if not isinstance(total_size, (int, float)) or total_size <= 0:
            raise ValueError(f"Invalid total size: {total_size}")
            
        if not isinstance(visible_size, (int, float)) or visible_size <= 0:
            raise ValueError(f"Invalid visible size: {visible_size}")
            
        if visible_size > total_size:
            raise ValueError("Visible size cannot exceed total size")
            
        if visible_size < self.executor.min_order_size:
            raise ValueError(f"Visible size below minimum {self.executor.min_order_size}")
            
        if price is not None and (not isinstance(price, (int, float)) or price <= 0):
            raise ValueError(f"Invalid price: {price}")
    
    def _initialize_execution_result(self, execution_id: str, symbol: str,
                                   side: str, total_size: float,
                                   visible_size: float) -> Dict:
        """Initialize iceberg execution result"""
        return {
            'execution_id': execution_id,
            'symbol': symbol,
            'side': side,
            'total_size': total_size,
            'visible_size': visible_size,
            'start_time': datetime.now(),
            'child_orders': [],
            'total_filled': 0,
            'average_price': 0,
            'status': 'running',
            'errors': []
        }
    
    async def _execute_iceberg_slices(self, execution_result: Dict, total_size: float,
                                    visible_size: float, price: Optional[float]):
        """Execute iceberg order slices"""
        remaining_size = total_size
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while remaining_size > self.executor.min_order_size and execution_result['execution_id'] in self.executor.active_orders:
            try:
                # Check failure threshold
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive failures in iceberg execution")
                    execution_result['status'] = 'failed'
                    break
                
                # Calculate current slice size
                current_slice = min(visible_size, remaining_size)
                
                # Place visible portion
                child_order = await self.executor._place_child_order_with_retries(
                    symbol=execution_result['symbol'],
                    side=execution_result['side'],
                    size=current_slice,
                    price=price,
                    execution_id=execution_result['execution_id']
                )
                
                # Process result
                if child_order and child_order.get('status') != 'error':
                    execution_result['child_orders'].append(child_order)
                    consecutive_failures = 0
                    
                    if child_order.get('status') in ['filled', 'partial']:
                        filled_size = child_order.get('filled_size', 0)
                        fill_price = child_order.get('fill_price', 0)
                        
                        if filled_size > 0 and fill_price > 0:
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
                else:
                    consecutive_failures += 1
                    execution_result['errors'].append(f"Failed iceberg slice at {datetime.now()}")
                
                # Dynamic wait time based on market conditions
                wait_time = self.executor._calculate_iceberg_wait_time(
                    execution_result['symbol'], execution_result['side'], consecutive_failures
                )
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in iceberg slice: {e}")
                execution_result['errors'].append(f"Iceberg error: {str(e)}")
                consecutive_failures += 1
                await asyncio.sleep(10)  # Wait longer on error
    
    def _finalize_execution(self, execution_result: Dict, total_size: float) -> Dict:
        """Finalize iceberg execution"""
        execution_result['end_time'] = datetime.now()
        
        if execution_result['status'] == 'running':
            execution_result['status'] = 'completed'
            self.executor.execution_metrics['successful_executions'] += 1
        else:
            self.executor.execution_metrics['failed_executions'] += 1
            
        execution_result['fill_rate'] = execution_result['total_filled'] / total_size
        
        self.executor._add_to_history(execution_result)
        
        if execution_result['execution_id'] in self.executor.active_orders:
            del self.executor.active_orders[execution_result['execution_id']]
        
        return execution_result
