"""
File: main.py
Modified: 2024-12-19
Enhanced: 2024-12-20
Refactored: 2025-07-18

Main entry point for the Hyperliquid Trading Bot.
This file has been refactored to use modular components for better maintainability.
"""

import asyncio
import signal
import sys
import os
import traceback
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Import modular components
from main.modules.health_monitoring import HealthMonitor
from main.modules.component_initializer import ComponentInitializer
from main.modules.config_validator import ConfigValidator
from main.modules.ml_predictor import MLPredictor, RLSignalGenerator
from main.modules.task_supervisor import TaskSupervisor, PerformanceTracker, ModelTrainer, ShutdownManager

# Import utilities
from utils.config import Config
from utils.logger import setup_logger, log_trade

logger = setup_logger(__name__)


class HyperliquidTradingBot:
    """Main trading bot orchestrator with comprehensive error handling"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize trading bot with error handling.
        
        Args:
            config_path: Path to configuration file
        """
        self.initialized = False
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        try:
            # Load and validate configuration
            logger.info(f"Loading configuration from {config_path}")
            self.config = Config(config_path)
            
            # Validate configuration
            config_validator = ConfigValidator(self.config)
            if not config_validator.validate():
                raise ValueError("Invalid configuration")
            
            # Log configuration summary
            config_summary = config_validator.get_config_summary()
            logger.info(f"Configuration summary: {json.dumps(config_summary, indent=2)}")
            
            # Initialize health monitoring
            self.health_monitor = HealthMonitor(self.config.to_dict())
            
            # Initialize components
            self._initialize_with_retries()
            
            # Set memory baseline after initialization
            self.health_monitor.memory_detector.set_baseline()
            
            self.initialized = True
            logger.info("Trading bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            logger.error(traceback.format_exc())
            self._cleanup_partial_initialization()
            raise
    
    def _initialize_with_retries(self, max_retries: int = 3):
        """Initialize components with retry logic"""
        for attempt in range(max_retries):
            try:
                self._initialize_components()
                return
            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 5
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    asyncio.get_event_loop().run_until_complete(asyncio.sleep(sleep_time))
                else:
                    raise
    
    def _initialize_components(self):
        """Initialize all bot components"""
        # Initialize component initializer
        self.component_initializer = ComponentInitializer(
            self.config.to_dict(),
            self.health_monitor.health_tracker
        )
        
        # Initialize all components
        components = asyncio.get_event_loop().run_until_complete(
            self.component_initializer.initialize_all_components()
        )
        
        # Store component references
        self.__dict__.update(components)
        
        # Initialize additional managers
        self._initialize_managers()
        
        # Initialize performance tracking
        self.performance = {
            'start_time': datetime.now(),
            'initial_capital': self.config.get('trading.initial_capital'),
            'trades': 0,
            'wins': 0,
            'total_pnl': 0,
            'errors': 0,
            'last_health_check': datetime.now(),
            'component_errors': defaultdict(int)
        }
    
    def _initialize_managers(self):
        """Initialize additional manager components"""
        # ML Predictor
        self.ml_predictor = MLPredictor(
            self.ensemble_predictor,
            self.rl_system,
            self.regime_detector
        )
        
        # RL Signal Generator
        self.rl_signal_generator = RLSignalGenerator(self.risk_manager)
        
        # Task Supervisor
        self.task_supervisor = TaskSupervisor(
            self.health_monitor.health_tracker,
            self.health_monitor.deadlock_detector
        )
        
        # Performance Tracker
        self.performance_tracker = PerformanceTracker(self.db)
        
        # Model Trainer
        self.model_trainer = ModelTrainer(
            self.config.to_dict(),
            self.rl_system,
            self.ensemble_predictor,
            self.db
        )
        
        # Shutdown Manager
        self.shutdown_manager = ShutdownManager(
            self.__dict__,
            self.config.to_dict()
        )
    
    def _cleanup_partial_initialization(self):
        """Clean up partially initialized components"""
        if hasattr(self, 'component_initializer'):
            asyncio.get_event_loop().run_until_complete(
                self.component_initializer.cleanup_partial_initialization()
            )
    
    async def start(self):
        """Start the trading bot with comprehensive error handling"""
        if not self.initialized:
            raise RuntimeError("Bot not properly initialized")
            
        logger.info("Starting Hyperliquid trading bot...")
        
        try:
            self.running = True
            self.task_supervisor.running = True
            
            # Connect to exchange with retries
            await self._connect_to_exchange()
            
            # Start data collection
            symbols = self.config.get('trading.symbols', ['BTC-USD', 'ETH-USD'])
            await self._start_data_collection(symbols)
            
            # Start dashboard if enabled
            if self.dashboard_manager and self.config.get('dashboard.enabled', True):
                try:
                    self.dashboard_manager.start_dashboard()
                except Exception as e:
                    logger.warning(f"Failed to start dashboard: {e}")
                    self.health_monitor.health_tracker.update_status('dashboard', 'failed', str(e))
            
            # Define all tasks
            task_definitions = self._get_task_definitions()
            
            # Start all tasks with supervision
            tasks = await self.task_supervisor.start_all_tasks(task_definitions)
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
            logger.error(traceback.format_exc())
            self.performance['errors'] += 1
        finally:
            await self.shutdown()
    
    async def _connect_to_exchange(self):
        """Connect to exchange with retries"""
        connected = False
        for attempt in range(3):
            try:
                await self.exchange.connect_websocket()
                connected = True
                break
            except Exception as e:
                logger.error(f"Failed to connect to exchange (attempt {attempt + 1}): {e}")
                self.health_monitor.increment_partition_count('exchange')
                if attempt < 2:
                    await asyncio.sleep(5)
                    
        if not connected:
            raise ConnectionError("Failed to connect to exchange after 3 attempts")
    
    def _get_task_definitions(self) -> List[tuple]:
        """Get task definitions for the supervisor"""
        # Core trading tasks
        core_tasks = [
            ('trading_loop', self._trading_loop(), True),
            ('risk_monitoring', self._risk_monitoring_loop(), True),
        ]
        
        # Optional tasks
        optional_tasks = [
            ('model_update', self.model_trainer.model_update_loop(), False),
            ('portfolio_rebalancing', self._portfolio_rebalancing_loop(), False),
            ('performance_tracking', self.performance_tracker.performance_tracking_loop(), False),
        ]
        
        # Monitoring tasks
        monitoring_tasks = [
            ('memory_monitor', self.health_monitor.memory_monitoring_loop(
                self.running, self._cleanup_memory_callback), False),
            ('deadlock_monitor', self.health_monitor.deadlock_monitoring_loop(
                self.running, self._restart_task_callback), False),
            ('component_health', self.health_monitor.component_health_loop(self.running), False)
        ]
        
        # Portfolio monitoring
        if self.portfolio_monitor:
            optional_tasks.append((
                'portfolio_monitoring',
                self.portfolio_monitor.start_monitoring(
                    self.risk_manager,
                    self.portfolio_analytics,
                    self.data_collector
                ),
                False
            ))
        
        # Health check
        optional_tasks.append(('health_check', self._health_check_loop(), False))
        
        return core_tasks + optional_tasks + monitoring_tasks
    
    async def _cleanup_memory_callback(self):
        """Callback for memory cleanup"""
        if hasattr(self, 'data_collector'):
            self.data_collector.clear_cache()
        if hasattr(self, 'ensemble_predictor'):
            self.ensemble_predictor.clear_cache()
    
    async def _restart_task_callback(self, task_name: str):
        """Callback to restart a deadlocked task"""
        # Implementation depends on specific task restart logic
        logger.info(f"Restarting task: {task_name}")
    
    async def _start_data_collection(self, symbols: List[str]):
        """Start data collection for all symbols"""
        for symbol in symbols:
            await self.data_collector.subscribe_symbol(symbol)
        
        # Wait for initial data
        await asyncio.sleep(5)
    
    async def _trading_loop(self):
        """Main trading loop"""
        await asyncio.sleep(5)  # Wait for initial data
        
        while self.running:
            try:
                for symbol in self.config.get('trading.symbols', ['BTC-USD', 'ETH-USD']):
                    # Get market data
                    market_data = await self._get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # Get ML predictions
                    ml_predictions = await self.ml_predictor.get_predictions(market_data)
                    
                    # Generate signals from each strategy
                    all_signals = []
                    
                    # Add RL-based signals
                    if ml_predictions.get('rl_action', 0) != 0:  # 0 is hold
                        rl_signal = self.rl_signal_generator.create_signal(
                            symbol,
                            ml_predictions['rl_action'],
                            ml_predictions['rl_confidence'],
                            market_data['current'],
                            ml_predictions.get('rl_metadata')
                        )
                        if rl_signal:
                            all_signals.append(rl_signal)
                    
                    # Add strategy signals
                    for strategy_name, strategy in self.strategies.items():
                        if strategy:
                            signal = strategy.generate_signal(market_data, ml_predictions)
                            if signal:
                                all_signals.append(signal)
                    
                    # Filter and rank signals
                    if all_signals:
                        best_signal = self.adaptive_strategy_manager.select_best_signal(
                            all_signals, market_data, ml_predictions
                        )
                        
                        if best_signal:
                            await self._execute_signal(best_signal, market_data)
                    
                # Sleep before next iteration
                await asyncio.sleep(self.config.get('trading.loop_interval', 1))
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                logger.error(traceback.format_exc())
                self.performance['errors'] += 1
                await asyncio.sleep(5)
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for a symbol"""
        try:
            # Get OHLCV data
            ohlcv_data = await self.data_collector.get_ohlcv(symbol, limit=500)
            if ohlcv_data is None or len(ohlcv_data) < 50:
                return None
            
            # Get current market data
            current_data = await self.data_collector.get_market_data(symbol)
            if not current_data:
                return None
            
            # Process data
            df = pd.DataFrame(ohlcv_data)
            df = self.preprocessor.prepare_ohlcv_data(df)
            df = self.preprocessor.calculate_technical_indicators(df)
            df = self.feature_engineer.engineer_all_features(df)
            
            return {
                'ohlcv': df,
                'current': current_data,
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _execute_signal(self, signal, market_data: Dict):
        """Execute a trading signal"""
        try:
            symbol = signal.symbol
            
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                signal, market_data, self.portfolio_analytics.get_portfolio_metrics()
            )
            
            if position_size == 0:
                logger.info(f"Position size is 0 for {symbol}, skipping signal")
                return
            
            # Risk checks
            if not self.risk_manager.check_risk_limits(symbol, position_size, signal.direction):
                logger.warning(f"Risk limits exceeded for {symbol}")
                return
            
            # Execute order
            order_result = await self.advanced_executor.execute_order(
                symbol=symbol,
                side='buy' if signal.direction > 0 else 'sell',
                size=position_size,
                order_type='limit',
                price=signal.entry_price,
                time_in_force='IOC'
            )
            
            if order_result and order_result.get('status') == 'filled':
                # Update risk manager
                self.risk_manager.update_position(
                    symbol=symbol,
                    size=position_size * signal.direction,
                    entry_price=order_result['fill_price']
                )
                
                # Log trade
                log_trade({
                    'symbol': symbol,
                    'side': 'buy' if signal.direction > 0 else 'sell',
                    'size': position_size,
                    'price': order_result['fill_price'],
                    'signal_type': signal.type,
                    'confidence': signal.confidence
                })
                
                # Update performance
                self.performance['trades'] += 1
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            self.performance['errors'] += 1
    
    async def _risk_monitoring_loop(self):
        """Monitor and manage risk continuously"""
        while self.running:
            try:
                # Check portfolio risk
                portfolio_metrics = self.portfolio_analytics.get_portfolio_metrics()
                
                # Check drawdown
                if portfolio_metrics['drawdown'] > self.config.get('risk.max_drawdown', 0.2):
                    logger.warning(f"Max drawdown exceeded: {portfolio_metrics['drawdown']:.2%}")
                    await self._reduce_all_positions()
                
                # Check individual position risks
                for symbol, position in self.risk_manager.positions.items():
                    current_price = await self._get_current_price(symbol)
                    if current_price:
                        position_pnl = self.risk_manager.calculate_position_pnl(
                            symbol, current_price
                        )
                        
                        # Stop loss check
                        if position_pnl < -self.config.get('risk.stop_loss', 0.02):
                            await self._close_position(symbol, "stop_loss")
                
                await asyncio.sleep(self.config.get('risk.monitoring_interval', 5))
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _portfolio_rebalancing_loop(self):
        """Periodically rebalance portfolio"""
        rebalance_interval = self.config.get('portfolio.rebalance_interval', 3600)  # 1 hour
        
        while self.running:
            try:
                # Get current portfolio state
                portfolio_state = self.portfolio_analytics.get_portfolio_state()
                
                # Calculate optimal weights using HRP
                returns_data = await self._get_returns_data()
                if returns_data is not None:
                    optimal_weights = self.hrp_optimizer.calculate_weights(returns_data)
                    
                    # Execute rebalancing if needed
                    rebalancing_trades = self._calculate_rebalancing_trades(
                        portfolio_state, optimal_weights
                    )
                    
                    for trade in rebalancing_trades:
                        await self._execute_rebalancing_trade(trade)
                
                await asyncio.sleep(rebalance_interval)
                
            except Exception as e:
                logger.error(f"Error in portfolio rebalancing: {e}")
                await asyncio.sleep(rebalance_interval)
    
    async def _health_check_loop(self):
        """Periodic health checks"""
        while self.running:
            try:
                # Check exchange connection
                if not await self.exchange.is_connected():
                    logger.warning("Exchange disconnected, attempting reconnection...")
                    await self._connect_to_exchange()
                
                # Check data feed
                for symbol in self.config.get('trading.symbols', []):
                    last_update = self.data_collector.get_last_update(symbol)
                    if last_update and (datetime.now() - last_update).total_seconds() > 60:
                        logger.warning(f"Data feed stale for {symbol}")
                
                # Update performance tracker
                self.performance_tracker.update_rl_metrics(self.rl_system)
                
                self.performance['last_health_check'] = datetime.now()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(30)
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            market_data = await self.data_collector.get_market_data(symbol)
            return market_data.get('mid_price') if market_data else None
        except Exception:
            return None
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.risk_manager.positions.get(symbol)
            if not position:
                return
            
            # Execute close order
            side = 'sell' if position['size'] > 0 else 'buy'
            size = abs(position['size'])
            
            order_result = await self.advanced_executor.execute_order(
                symbol=symbol,
                side=side,
                size=size,
                order_type='market'
            )
            
            if order_result and order_result.get('status') == 'filled':
                # Calculate PnL
                pnl = self.risk_manager.close_position(symbol, order_result['fill_price'])
                
                # Update performance
                self.performance['total_pnl'] += pnl
                if pnl > 0:
                    self.performance['wins'] += 1
                
                # Update performance tracker
                self.performance_tracker.update_trade_metrics({
                    'pnl': pnl,
                    'reason': reason
                })
                
                logger.info(f"Closed position {symbol} for {reason}, PnL: ${pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
    
    async def _reduce_all_positions(self):
        """Reduce all positions due to risk limit"""
        for symbol in list(self.risk_manager.positions.keys()):
            await self._close_position(symbol, "risk_reduction")
    
    async def _get_returns_data(self) -> Optional[pd.DataFrame]:
        """Get returns data for portfolio optimization"""
        try:
            symbols = self.config.get('trading.symbols', [])
            returns_list = []
            
            for symbol in symbols:
                df = await self.data_collector.get_ohlcv(symbol, limit=100)
                if df is not None and len(df) > 20:
                    returns = pd.DataFrame(df)['close'].pct_change().dropna()
                    returns_list.append(returns.rename(symbol))
            
            if returns_list:
                return pd.concat(returns_list, axis=1)
            return None
            
        except Exception as e:
            logger.error(f"Error getting returns data: {e}")
            return None
    
    def _calculate_rebalancing_trades(self, current_state: Dict, 
                                    optimal_weights: Dict) -> List[Dict]:
        """Calculate trades needed for rebalancing"""
        trades = []
        
        # Implementation depends on your specific rebalancing logic
        # This is a simplified example
        
        return trades
    
    async def _execute_rebalancing_trade(self, trade: Dict):
        """Execute a rebalancing trade"""
        # Implementation depends on your specific execution logic
        pass
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        self.running = False
        await self.shutdown_manager.shutdown()
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}")
        self.shutdown_event.set()
        # Create shutdown task in the event loop
        asyncio.create_task(self.shutdown())


async def main():
    """Main entry point with error handling"""
    bot = None
    
    try:
        # Create bot instance
        bot = HyperliquidTradingBot()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, bot.signal_handler)
        signal.signal(signal.SIGTERM, bot.signal_handler)
        
        # Handle Windows signals if on Windows
        if sys.platform == 'win32':
            signal.signal(signal.SIGBREAK, bot.signal_handler)
        
        # Start bot
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if bot:
            await bot.shutdown()


if __name__ == "__main__":
    # Setup asyncio error handler
    def exception_handler(loop, context):
        exception = context.get('exception')
        if isinstance(exception, KeyboardInterrupt):
            return
        logger.error(f"Unhandled exception in event loop: {context}")
    
    # Configure asyncio for production
    if sys.platform == 'win32':
        # Windows specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(exception_handler)
    
    # Set process priority if possible
    try:
        import os
        if hasattr(os, 'nice'):
            os.nice(-5)  # Increase priority slightly
    except Exception:
        pass
    
    try:
        loop.run_until_complete(main())
    except Exception as e:
        logger.critical(f"Failed to run main: {e}")
        sys.exit(1)
    finally:
        loop.close()

"""
REFACTORING SUMMARY:
- Original file: 1700+ lines
- Refactored main.py: ~600 lines
- Created 5 modular components:
  1. health_monitoring.py - Health tracking, memory, and deadlock detection
  2. component_initializer.py - Component initialization logic
  3. config_validator.py - Configuration validation
  4. ml_predictor.py - ML prediction logic including RL
  5. task_supervisor.py - Task management, performance tracking, shutdown
- Benefits:
  * Better separation of concerns
  * Easier to test individual components
  * More maintainable code structure
  * Clearer error handling boundaries
  * Reduced complexity in main file
"""
