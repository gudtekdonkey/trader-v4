"""
File: market_making.py (Refactored)
Modified: 2024-12-20
Changes Summary:
- Modularized into 5 separate modules for better organization
- Maintained all error handling and validation logic
- Improved separation of concerns
- No change in external API or functionality
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import traceback

from ...risk_manager.risk_manager import RiskManager
from ....utils.logger import setup_logger

# Import modular components
from .modules.quote_generator import QuoteGenerator, MarketMakingQuote
from .modules.inventory_manager import InventoryManager
from .modules.order_manager import OrderManager
from .modules.performance_tracker import PerformanceTracker
from .modules.risk_controller import RiskController

logger = setup_logger(__name__)


class MarketMakingStrategy:
    """Advanced market making strategy with inventory management and comprehensive error handling"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        
        # Strategy parameters with validation
        self.params = {
            'base_spread': 0.002,  # 0.2% base spread
            'min_spread': 0.001,   # 0.1% minimum spread
            'max_spread': 0.01,    # 1% maximum spread
            'inventory_target': 0,  # Target neutral inventory
            'max_inventory': 10000,  # Maximum inventory in USD
            'inventory_skew_factor': 0.5,  # How much to skew quotes based on inventory
            'volatility_factor': 2.0,  # Spread multiplier for volatility
            'order_levels': 3,  # Number of order levels
            'level_spacing': 0.001,  # 0.1% between levels
            'size_decay': 0.5,  # Size reduction per level
            'min_order_size': 10,  # Minimum order size in USD
            'rebalance_threshold': 0.1,  # 10% inventory imbalance triggers rebalance
            'maker_fee': -0.0001,  # -0.01% maker rebate
            'taker_fee': 0.00035,   # 0.035% taker fee
            'max_quote_failures': 10,  # Maximum consecutive quote failures
            'emergency_spread_multiplier': 3.0  # Spread multiplier in emergency mode
        }
        
        # Initialize modules
        self.risk_controller = RiskController(self.params)
        self.quote_generator = QuoteGenerator(self.params)
        self.inventory_manager = InventoryManager(self.params)
        self.order_manager = OrderManager()
        self.performance_tracker = PerformanceTracker()
        
        # Validate parameters
        if not self.risk_controller.validate_parameters():
            raise ValueError("Invalid market making parameters")
    
    def calculate_quotes(self, market_data: Dict[str, any], 
                        ml_predictions: Optional[Dict] = None) -> List[MarketMakingQuote]:
        """Calculate market making quotes with comprehensive error handling"""
        quotes = []
        
        # Check emergency stop
        if self.risk_controller.emergency_stop:
            logger.warning("Emergency stop active, no quotes generated")
            return []
        
        # Check circuit breaker
        if self.risk_controller.check_circuit_breaker():
            logger.warning("Circuit breaker active, no quotes generated")
            return []
        
        try:
            # Validate market conditions
            if not self.risk_controller.validate_market_conditions(market_data):
                logger.error("Invalid market conditions")
                self.risk_controller.record_quote_failure(market_data.get('symbol', 'UNKNOWN'))
                return []
            
            # Get market state
            symbol = market_data.get('symbol', 'BTC-USD')
            best_bid = market_data['best_bid']
            best_ask = market_data['best_ask']
            mid_price = (best_bid + best_ask) / 2
            current_spread = best_ask - best_bid
            
            # Calculate fair value
            fair_value = self.quote_generator.calculate_fair_value(market_data, ml_predictions)
            
            # Get inventory position
            current_inventory = self.inventory_manager.inventory.get(symbol, 0)
            inventory_value = current_inventory * mid_price
            
            # Check inventory limits
            if not self.risk_controller.check_inventory_limits(symbol, inventory_value):
                # Generate emergency quotes
                emergency_quote = self.quote_generator.generate_emergency_quote(
                    symbol, market_data, current_inventory, mid_price
                )
                if emergency_quote and self.quote_generator.validate_quote(emergency_quote, market_data):
                    return [emergency_quote]
                return []
            
            # Calculate inventory skew
            inventory_skew = self.inventory_manager.calculate_inventory_skew(symbol, mid_price)
            
            # Get volatility
            volatility = market_data.get('volatility', 0.02)
            if not 0 < volatility < 1:
                logger.warning(f"Invalid volatility: {volatility}, using default")
                volatility = 0.02
            
            # Get available capital
            available_capital = self.risk_manager.get_available_capital()
            
            # Generate quotes for each level
            for level in range(min(self.params['order_levels'], 5)):
                try:
                    # Calculate base size with risk adjustment
                    risk_adjustment = self.risk_controller.get_risk_adjustment_factor()
                    base_size = self.risk_controller.calculate_base_size(
                        available_capital * risk_adjustment, market_data, level
                    )
                    
                    if base_size <= 0:
                        continue
                    
                    # Generate quote
                    quote = self.quote_generator.generate_quote(
                        level, symbol, fair_value, current_spread,
                        volatility, inventory_skew, base_size,
                        market_data, ml_predictions
                    )
                    
                    if quote and self.quote_generator.validate_quote(quote, market_data):
                        quotes.append(quote)
                    else:
                        logger.warning(f"Invalid quote generated for level {level}")
                        
                except Exception as e:
                    logger.error(f"Error generating quote for level {level}: {e}")
                    self.performance_tracker.performance['errors'] += 1
                    continue
            
            # Update microstructure tracking
            self.performance_tracker.update_microstructure(symbol, market_data)
            
            # Record success/failure
            if quotes:
                self.risk_controller.record_quote_success()
            else:
                self.risk_controller.record_quote_failure(symbol)
                
            return quotes
            
        except Exception as e:
            logger.error(f"Critical error in calculate_quotes: {e}")
            logger.error(traceback.format_exc())
            self.risk_controller.record_quote_failure(market_data.get('symbol', 'UNKNOWN'))
            return []
    
    async def execute_quotes(self, quotes: List[MarketMakingQuote], executor) -> Dict:
        """Place market making orders with error handling"""
        try:
            results = await self.order_manager.place_quotes(quotes, executor)
            
            # Update performance metrics
            self.performance_tracker.update_quote_metrics(
                len(results['placed_orders']),
                len(results['failed_orders'])
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in execute_quotes: {e}")
            logger.error(traceback.format_exc())
            self.performance_tracker.performance['errors'] += 1
            return {
                'placed_orders': [],
                'failed_orders': [],
                'total_quotes': len(quotes) * 2
            }
    
    def handle_fill(self, fill_data: Dict):
        """Handle order fill with comprehensive error handling"""
        try:
            # Update inventory
            if not self.inventory_manager.update_inventory(fill_data):
                return
            
            # Update performance metrics
            self.performance_tracker.update_fill_metrics(fill_data)
            
            # Calculate spread captured
            symbol = fill_data['symbol']
            side = fill_data['side']
            price = float(fill_data['price'])
            size = float(fill_data['size'])
            
            spread_captured = self.inventory_manager.update_spread_captured(
                symbol, side, price, size
            )
            if spread_captured > 0:
                self.performance_tracker.update_spread_captured(spread_captured)
            
            # Check if rebalancing needed
            current_price = price  # Use fill price as estimate
            if self.inventory_manager.needs_rebalancing(symbol, current_price):
                logger.info(f"Inventory rebalancing needed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error handling fill: {e}")
            logger.error(traceback.format_exc())
            self.performance_tracker.performance['errors'] += 1
    
    def get_rebalance_orders(self, symbol: str, target_inventory: float = 0) -> List[Dict]:
        """Generate orders to rebalance inventory"""
        try:
            # Get current price estimate
            recent_fills = self.inventory_manager.get_recent_fills(symbol, limit=10)
            if recent_fills:
                current_price = sum(f['price'] for f in recent_fills) / len(recent_fills)
            else:
                return []
            
            return self.inventory_manager.generate_rebalance_orders(
                symbol, current_price, target_inventory
            )
            
        except Exception as e:
            logger.error(f"Error generating rebalance orders: {e}")
            return []
    
    def calculate_inventory_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized PnL"""
        pnl_by_symbol = self.inventory_manager.calculate_inventory_pnl(current_prices)
        total_pnl = pnl_by_symbol.get('total', 0)
        self.performance_tracker.performance['inventory_pnl'] = total_pnl
        return total_pnl
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.risk_controller.reset_emergency_stop()
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy performance metrics with error handling"""
        try:
            # Get performance summary
            performance_summary = self.performance_tracker.get_performance_summary()
            
            # Get inventory metrics
            inventory_metrics = self.inventory_manager.get_inventory_metrics()
            
            # Calculate additional metrics
            performance_summary['inventory_turnover'] = self.performance_tracker.calculate_inventory_turnover(
                inventory_metrics
            )
            
            return {
                'name': 'Market Making',
                'performance': performance_summary,
                'inventory': inventory_metrics,
                'active_quotes': self.order_manager.get_active_orders_count(),
                'order_metrics': self.order_manager.get_order_metrics(),
                'parameters': self.params.copy(),
                'health': self.risk_controller.get_risk_status(),
                'time_series': {
                    'hourly': self.performance_tracker.get_time_series_metrics('hourly'),
                    'daily': self.performance_tracker.get_time_series_metrics('daily')
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy metrics: {e}")
            return {
                'name': 'Market Making',
                'error': str(e)
            }
    
    async def shutdown(self, executor):
        """Graceful shutdown of strategy"""
        try:
            logger.info("Shutting down market making strategy")
            
            # Cancel all active orders
            await self.order_manager.cancel_all_orders(executor)
            
            # Save final performance snapshot
            current_prices = {}
            for symbol in self.inventory_manager.inventory.keys():
                recent_fills = self.inventory_manager.get_recent_fills(symbol, limit=1)
                if recent_fills:
                    current_prices[symbol] = recent_fills[0]['price']
            
            inventory_pnl = self.calculate_inventory_pnl(current_prices)
            self.performance_tracker.save_hourly_snapshot(inventory_pnl)
            
            logger.info("Market making strategy shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

"""
MODULARIZATION SUMMARY:
- Original file: 1,500+ lines
- Refactored into 5 modules:
  1. quote_generator.py (~400 lines) - Quote generation and fair value calculation
  2. inventory_manager.py (~350 lines) - Position tracking and rebalancing
  3. order_manager.py (~250 lines) - Order placement and tracking
  4. performance_tracker.py (~400 lines) - Metrics and analytics
  5. risk_controller.py (~300 lines) - Risk controls and circuit breakers
- Main file reduced to ~300 lines
- All error handling preserved
- No breaking changes to external API
- Improved testability and maintainability
"""