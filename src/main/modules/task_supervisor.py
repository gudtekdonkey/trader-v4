"""
Task supervision module for the trading bot.
Handles task creation, monitoring, and error recovery.
"""

import asyncio
import traceback
from typing import Dict, Optional, Callable, Any
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger(__name__)


class TaskSupervisor:
    """Supervises and manages async tasks with error handling"""
    
    def __init__(self, health_tracker=None, deadlock_detector=None):
        self.health_tracker = health_tracker
        self.deadlock_detector = deadlock_detector
        self.running = True
        self.critical_error_count = 0
        self.max_critical_errors = 5
        self.performance_metrics = {
            'errors': 0,
            'component_errors': {}
        }
        self.shutdown_event = asyncio.Event()
        
    async def create_supervised_task(self, name: str, coro, critical: bool = False):
        """Create a supervised task that handles errors"""
        try:
            # Register with deadlock detector
            if self.deadlock_detector:
                task = asyncio.current_task()
                if task:
                    self.deadlock_detector.register_task(name, task)
            
            await coro
            
        except asyncio.CancelledError:
            logger.info(f"Task {name} cancelled")
            raise
            
        except Exception as e:
            logger.error(f"Error in {name} task: {e}")
            logger.error(traceback.format_exc())
            
            self.performance_metrics['errors'] += 1
            if name not in self.performance_metrics['component_errors']:
                self.performance_metrics['component_errors'][name] = 0
            self.performance_metrics['component_errors'][name] += 1
            
            if critical:
                self.critical_error_count += 1
                if self.critical_error_count >= self.max_critical_errors:
                    logger.critical(f"Too many critical errors ({self.critical_error_count}), shutting down")
                    self.running = False
                    self.shutdown_event.set()
                else:
                    logger.warning(f"Critical error {self.critical_error_count}/{self.max_critical_errors}")
                    # Restart critical task after delay
                    await asyncio.sleep(30)
                    if self.running:
                        return await self.create_supervised_task(name, coro, critical)
            else:
                # Update component health
                if self.health_tracker:
                    self.health_tracker.update_status(name, 'failed', str(e))
                    
        finally:
            # Unregister from deadlock detector
            if self.deadlock_detector:
                self.deadlock_detector.unregister_task(name)
    
    async def start_all_tasks(self, task_definitions: list) -> list:
        """Start all tasks with supervision"""
        tasks = []
        
        for name, task_coro, is_critical in task_definitions:
            task = asyncio.create_task(
                self.create_supervised_task(name, task_coro, critical=is_critical)
            )
            tasks.append(task)
            
        return tasks
    
    def stop_all_tasks(self):
        """Signal all tasks to stop"""
        self.running = False
        self.shutdown_event.set()


class PerformanceTracker:
    """Tracks and reports performance metrics"""
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.start_time = datetime.now()
        self.metrics = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'errors': 0,
            'last_update': datetime.now()
        }
        self.rl_metrics = {}
        
    def update_trade_metrics(self, trade_result: Dict):
        """Update metrics after a trade"""
        self.metrics['trades'] += 1
        
        pnl = trade_result.get('pnl', 0)
        self.metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.metrics['wins'] += 1
        else:
            self.metrics['losses'] += 1
            
        self.metrics['last_update'] = datetime.now()
        
        # Save to database if available
        if self.db:
            try:
                self.db.record_trade_metrics(self.metrics)
            except Exception as e:
                logger.error(f"Error saving trade metrics: {e}")
    
    def update_rl_metrics(self, rl_system):
        """Update RL agent performance metrics"""
        try:
            metrics = {}
            
            # Get recent performance for each agent
            for agent_name, performance_history in rl_system.agent_performance.items():
                if len(performance_history) >= 10:
                    recent_perf = performance_history[-10:]
                    avg_reward = sum(recent_perf) / len(recent_perf)
                    metrics[f'{agent_name}_perf'] = f"{avg_reward:.2f}"
                else:
                    metrics[f'{agent_name}_perf'] = "No data"
            
            # Calculate average confidence across agents
            total_confidence = 0
            count = 0
            
            for agent_name, agent in rl_system.agents.items():
                if agent is not None:
                    try:
                        # Get confidence on a dummy state
                        import numpy as np
                        dummy_state = np.zeros(75)
                        confidence = agent.get_confidence(dummy_state)
                        total_confidence += confidence
                        count += 1
                    except Exception:
                        pass
            
            metrics['avg_confidence'] = total_confidence / count if count > 0 else 0
            
            self.rl_metrics = metrics
            
        except Exception as e:
            logger.error(f"Error calculating RL metrics: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600  # hours
        
        win_rate = self.metrics['wins'] / self.metrics['trades'] if self.metrics['trades'] > 0 else 0
        avg_pnl = self.metrics['total_pnl'] / self.metrics['trades'] if self.metrics['trades'] > 0 else 0
        
        return {
            'runtime_hours': runtime,
            'total_trades': self.metrics['trades'],
            'win_rate': win_rate,
            'total_pnl': self.metrics['total_pnl'],
            'avg_pnl_per_trade': avg_pnl,
            'errors': self.metrics['errors'],
            'rl_metrics': self.rl_metrics
        }
    
    async def performance_tracking_loop(self, interval: int = 300):
        """Periodically log performance metrics"""
        while True:
            try:
                summary = self.get_performance_summary()
                
                logger.info(f"""
Performance Summary:
- Runtime: {summary['runtime_hours']:.2f} hours
- Total Trades: {summary['total_trades']}
- Win Rate: {summary['win_rate']:.2%}
- Total PnL: ${summary['total_pnl']:.2f}
- Avg PnL/Trade: ${summary['avg_pnl_per_trade']:.2f}
- Errors: {summary['errors']}
                """.strip())
                
                # Log RL metrics if available
                if summary['rl_metrics']:
                    logger.info("RL Agent Performance:")
                    for metric, value in summary['rl_metrics'].items():
                        logger.info(f"- {metric}: {value}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(interval)


class ModelTrainer:
    """Handles periodic model training and updates"""
    
    def __init__(self, config: Dict, rl_system=None, ensemble_predictor=None, db_manager=None):
        self.config = config
        self.rl_system = rl_system
        self.ensemble_predictor = ensemble_predictor
        self.db = db_manager
        
    async def model_update_loop(self, interval: int = 86400):  # Daily by default
        """Update ML models periodically"""
        while True:
            try:
                logger.info("Starting model update...")
                
                # Update ensemble models
                if self.ensemble_predictor:
                    await self._update_ensemble_models()
                
                # Train RL agents if enabled
                if self.config.get('ml_models.train_rl_agents', False) and self.rl_system:
                    await self._train_rl_agents()
                
                logger.info("Model update completed")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in model update: {e}")
                await asyncio.sleep(interval)
    
    async def _update_ensemble_models(self):
        """Update ensemble models with recent data"""
        try:
            # Implementation depends on your ensemble predictor
            logger.info("Updating ensemble models...")
            
            # Load recent data
            if self.db:
                # Get data for all symbols
                symbols = self.config.get('trading.symbols', ['BTC-USD'])
                
                for symbol in symbols:
                    # Load recent data
                    from datetime import timedelta
                    df = self.db.load_market_data(
                        symbol,
                        start_date=datetime.now() - timedelta(days=30)
                    )
                    
                    if len(df) > 1000:
                        # Update models with new data
                        self.ensemble_predictor.update_models(df)
                        
            logger.info("Ensemble models updated")
            
        except Exception as e:
            logger.error(f"Error updating ensemble models: {e}")
    
    async def _train_rl_agents(self):
        """Train multi-agent RL system"""
        try:
            logger.info("Starting RL agent training...")
            
            # Get training data
            symbols = self.config.get('trading.symbols', ['BTC-USD'])
            
            for symbol in symbols:
                # Load historical data
                from datetime import timedelta
                df = self.db.load_market_data(
                    symbol,
                    start_date=datetime.now() - timedelta(days=180)  # 6 months
                )
                
                if len(df) < 5000:  # Need substantial data for RL
                    logger.warning(f"Insufficient data for RL training on {symbol}")
                    continue
                
                # Prepare features
                from src.data.preprocessor import DataPreprocessor
                from src.data.feature_engineer import FeatureEngineer
                
                preprocessor = DataPreprocessor()
                feature_engineer = FeatureEngineer()
                
                df = preprocessor.prepare_ohlcv_data(df)
                df = preprocessor.calculate_technical_indicators(df)
                df = feature_engineer.engineer_all_features(df)
                
                # Train the multi-agent system
                training_episodes = self.config.get('ml_models.rl_training_episodes', 1000)
                self.rl_system.train_agents(
                    market_data=df,
                    training_episodes=training_episodes,
                    save_interval=100
                )
                
                # Save trained models
                save_path = self.config.get('ml_models.rl_models_path', 'models/rl_agents/')
                self.rl_system.save_all_agents(save_path)
                
                logger.info(f"Completed RL training for {symbol}")
                
        except Exception as e:
            logger.error(f"Error training RL agents: {e}")


class ShutdownManager:
    """Manages graceful shutdown of the trading bot"""
    
    def __init__(self, components: Dict, config: Dict):
        self.components = components
        self.config = config
        
    async def shutdown(self, reason: str = 'manual'):
        """Gracefully shutdown the bot with comprehensive cleanup"""
        logger.info(f"Initiating graceful shutdown (reason: {reason})...")
        
        # Save shutdown reason and metrics
        shutdown_info = {
            'timestamp': datetime.now(),
            'reason': reason,
            'memory_usage_mb': self._get_memory_usage()
        }
        
        # Save performance metrics if available
        if 'performance_tracker' in self.components:
            shutdown_info['performance'] = self.components['performance_tracker'].get_performance_summary()
        
        # Save component health if available
        if 'health_tracker' in self.components:
            shutdown_info['component_health'] = dict(self.components['health_tracker'].component_status)
        
        try:
            # Save shutdown info
            import json
            with open('logs/shutdown_info.json', 'w') as f:
                json.dump(shutdown_info, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save shutdown info: {e}")
        
        # Shutdown sequence
        await self._stop_monitoring()
        await self._cancel_orders()
        await self._close_positions()
        await self._save_state()
        await self._save_models()
        await self._close_connections()
        await self._final_cleanup()
        
        logger.info("Trading bot shutdown complete")
    
    async def _stop_monitoring(self):
        """Stop all monitoring tasks"""
        try:
            # Stop portfolio monitoring
            if 'portfolio_monitor' in self.components and self.components['portfolio_monitor']:
                self.components['portfolio_monitor'].stop_monitoring()
                
            # Stop dashboard
            if 'dashboard_manager' in self.components and self.components['dashboard_manager']:
                self.components['dashboard_manager'].stop_dashboard()
                
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    async def _cancel_orders(self):
        """Cancel all open orders"""
        try:
            if 'order_executor' in self.components:
                await self.components['order_executor'].cancel_all_orders()
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    async def _close_positions(self):
        """Close all open positions"""
        try:
            if 'risk_manager' in self.components:
                risk_manager = self.components['risk_manager']
                if risk_manager.positions:
                    for symbol in list(risk_manager.positions.keys()):
                        # Implementation depends on your close_position method
                        logger.info(f"Closing position for {symbol}")
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    async def _save_state(self):
        """Save final state to database"""
        try:
            if 'db' in self.components:
                db = self.components['db']
                
                final_state = {
                    'shutdown_time': datetime.now(),
                    'config': self.config
                }
                
                # Add component states
                if 'performance_tracker' in self.components:
                    final_state['performance'] = self.components['performance_tracker'].get_performance_summary()
                
                if 'health_tracker' in self.components:
                    final_state['health'] = {
                        'component_status': dict(self.components['health_tracker'].component_status),
                        'failure_counts': dict(self.components['health_tracker'].failure_counts)
                    }
                
                db.save_final_state(final_state)
                db.cleanup_old_data(90)  # Clean up data older than 90 days
                
        except Exception as e:
            logger.error(f"Error saving final state: {e}")
    
    async def _save_models(self):
        """Save ML model states"""
        try:
            # Save ensemble models
            if 'ensemble_predictor' in self.components:
                self.components['ensemble_predictor'].save_models('models/ensemble_shutdown')
                
            # Save RL models
            if 'rl_system' in self.components:
                self.components['rl_system'].save_all_agents('models/rl_agents_shutdown')
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def _close_connections(self):
        """Close all external connections"""
        try:
            # Close data collector
            if 'data_collector' in self.components:
                await self.components['data_collector'].close()
                
            # Close exchange connection
            if 'exchange' in self.components:
                await self.components['exchange'].close()
                
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    async def _final_cleanup(self):
        """Perform final memory cleanup"""
        try:
            # Clear caches and force garbage collection
            if 'memory_detector' in self.components:
                self.components['memory_detector'].cleanup_memory(aggressive=True)
            else:
                import gc
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in final cleanup: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except:
            return 0
