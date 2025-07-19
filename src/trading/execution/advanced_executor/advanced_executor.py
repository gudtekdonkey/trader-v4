"""
Advanced Order Executor - Main coordination module
Provides sophisticated order execution algorithms including TWAP, VWAP, and Iceberg orders.

File: advanced_executor.py
Modified: 2024-12-19
Refactored: 2025-01-18
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback

# Import modularized components
from .modules.execution_algorithms import TWAPExecutor, VWAPExecutor, IcebergExecutor
from .modules.order_manager import OrderManager
from .modules.volume_profile import VolumeProfileManager
from .modules.execution_analytics import ExecutionAnalytics

logger = logging.getLogger(__name__)


class AdvancedOrderExecutor:
    """
    Advanced order execution with TWAP, VWAP, and other algorithms.
    
    This is the main coordination class that delegates to specialized modules:
    - execution_algorithms: TWAP, VWAP, and Iceberg execution logic
    - order_manager: Child order placement and tracking
    - volume_profile: Volume data processing for VWAP
    - execution_analytics: Performance tracking and statistics
    """
    
    def __init__(self, exchange_client):
        """
        Initialize advanced order executor.
        
        Args:
            exchange_client: Exchange client for order placement
        """
        # Validate exchange client
        if not exchange_client:
            raise ValueError("Exchange client is required")
            
        self.exchange = exchange_client
        self.active_orders = {}
        self.execution_history = []
        
        # Execution limits
        self.max_active_executions = 10
        self.max_execution_history = 1000
        self.max_slice_retries = 3
        self.min_order_size = 0.0001  # Minimum order size
        self.max_execution_time_hours = 24  # Maximum execution time
        
        # Performance tracking
        self.execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'partial_executions': 0
        }
        
        # Initialize module managers
        self.twap_executor = TWAPExecutor(self)
        self.vwap_executor = VWAPExecutor(self)
        self.iceberg_executor = IcebergExecutor(self)
        self.order_manager = OrderManager(self)
        self.volume_manager = VolumeProfileManager(self)
        self.analytics = ExecutionAnalytics(self)
        
        logger.info("Advanced Order Executor initialized")
    
    # TWAP Execution
    async def execute_twap_order(self, 
                               symbol: str,
                               side: str,
                               total_size: float,
                               duration_minutes: int,
                               metadata: Optional[Dict] = None) -> Dict:
        """
        Execute TWAP (Time-Weighted Average Price) order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_size: Total size to execute
            duration_minutes: Duration over which to execute
            metadata: Additional order metadata
            
        Returns:
            Execution result dictionary
        """
        return await self.twap_executor.execute(
            symbol=symbol,
            side=side,
            total_size=total_size,
            duration_minutes=duration_minutes,
            metadata=metadata
        )
    
    # VWAP Execution
    async def execute_vwap_order(self,
                               symbol: str,
                               side: str,
                               total_size: float,
                               duration_minutes: int,
                               historical_volume_window: int = 20) -> Dict:
        """
        Execute VWAP (Volume-Weighted Average Price) order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_size: Total size to execute
            duration_minutes: Duration over which to execute
            historical_volume_window: Days of historical volume for weighting
            
        Returns:
            Execution result dictionary
        """
        return await self.vwap_executor.execute(
            symbol=symbol,
            side=side,
            total_size=total_size,
            duration_minutes=duration_minutes,
            historical_volume_window=historical_volume_window
        )
    
    # Iceberg Execution
    async def execute_iceberg_order(self,
                                  symbol: str,
                                  side: str,
                                  total_size: float,
                                  visible_size: float,
                                  price: Optional[float] = None) -> Dict:
        """
        Execute iceberg order (large order hidden by showing only small portions).
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_size: Total order size
            visible_size: Size visible in order book at any time
            price: Limit price (None for market price)
            
        Returns:
            Execution result dictionary
        """
        return await self.iceberg_executor.execute(
            symbol=symbol,
            side=side,
            total_size=total_size,
            visible_size=visible_size,
            price=price
        )
    
    # Order Management Methods (delegated to order_manager)
    async def _place_child_order_with_retries(self, **kwargs) -> Optional[Dict]:
        """Place child order with retry logic"""
        return await self.order_manager.place_child_order_with_retries(**kwargs)
    
    async def _place_child_order(self, **kwargs) -> Optional[Dict]:
        """Place individual child order"""
        return await self.order_manager.place_child_order(**kwargs)
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel active execution algorithm"""
        return self.order_manager.cancel_execution(execution_id)
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get status of active or completed execution"""
        return self.order_manager.get_execution_status(execution_id)
    
    def get_active_executions(self) -> List[Dict]:
        """Get list of all active executions"""
        return self.order_manager.get_active_executions()
    
    async def emergency_cancel_all(self) -> Dict[str, int]:
        """Emergency cancel all active executions"""
        return await self.order_manager.emergency_cancel_all()
    
    # Volume Profile Methods (delegated to volume_manager)
    async def _get_volume_profile_safe(self, symbol: str, days: int = 20) -> Optional[List[Dict]]:
        """Get historical volume profile for VWAP calculation"""
        return await self.volume_manager.get_volume_profile_safe(symbol, days)
    
    def _process_volume_data(self, historical_data: List[Dict]) -> List[Dict]:
        """Process raw volume data into hourly profile"""
        return self.volume_manager.process_volume_data(historical_data)
    
    def _calculate_vwap_schedule(self, total_size: float, duration_minutes: int, 
                               volume_profile: List[Dict]) -> List[Dict]:
        """Calculate VWAP execution schedule based on volume profile"""
        return self.volume_manager.calculate_vwap_schedule(
            total_size, duration_minutes, volume_profile
        )
    
    def _calculate_iceberg_wait_time(self, symbol: str, side: str, failures: int) -> float:
        """Calculate dynamic wait time for iceberg orders"""
        return self.volume_manager.calculate_iceberg_wait_time(symbol, side, failures)
    
    # Analytics Methods (delegated to analytics)
    def _add_to_history(self, execution_result: Dict):
        """Add execution to history with size limit"""
        self.analytics.add_to_history(execution_result)
    
    def get_execution_statistics(self) -> Dict:
        """Get execution performance statistics"""
        return self.analytics.get_execution_statistics()
    
    def get_execution_report(self, execution_id: str) -> Optional[Dict]:
        """Get detailed report for a specific execution"""
        return self.analytics.get_execution_report(execution_id)
    
    def get_symbol_statistics(self, symbol: str) -> Dict:
        """Get execution statistics for a specific symbol"""
        return self.analytics.get_symbol_statistics(symbol)


# Usage example with error handling
if __name__ == '__main__':
    import asyncio
    
    async def test_advanced_executor():
        # Mock exchange client
        class MockExchange:
            async def get_ticker(self, symbol):
                return {'bid': 50000, 'ask': 50010, 'last': 50005}
            
            async def place_order(self, **kwargs):
                return {'id': f"order_{int(time.time())}", 'status': 'placed'}
            
            async def get_order_status(self, order_id, symbol):
                # Simulate order fill
                return {
                    'id': order_id,
                    'status': 'filled',
                    'filled': 0.98,  # 98% fill
                    'average': 50005
                }
            
            async def get_historical_volume(self, symbol, days):
                return []  # Will trigger fallback
        
        # Test with error handling
        executor = AdvancedOrderExecutor(MockExchange())
        
        try:
            # Test TWAP execution
            result = await executor.execute_twap_order(
                symbol="BTC-USD",
                side="buy",
                total_size=1.0,
                duration_minutes=30
            )
            print(f"TWAP Result: {result}")
            
            # Test with invalid inputs
            try:
                await executor.execute_twap_order(
                    symbol="",  # Invalid symbol
                    side="buy",
                    total_size=1.0,
                    duration_minutes=30
                )
            except ValueError as e:
                print(f"Caught expected error: {e}")
            
            # Get statistics
            stats = executor.get_execution_statistics()
            print(f"Execution Statistics: {stats}")
            
        except Exception as e:
            print(f"Test error: {e}")
            traceback.print_exc()
    
    # Run test
    asyncio.run(test_advanced_executor())

"""
MODULARIZATION SUMMARY:
- Original file: 1,400+ lines
- Refactored main file: ~300 lines
- Modules created:
  1. execution_algorithms.py (~600 lines) - TWAP, VWAP, and Iceberg algorithms
  2. order_manager.py (~350 lines) - Order placement and tracking
  3. volume_profile.py (~250 lines) - Volume data processing
  4. execution_analytics.py (~300 lines) - Performance tracking and statistics
  
Benefits:
- Clear separation of concerns
- Easier to test individual components
- Better maintainability
- Modular algorithm implementations
"""
