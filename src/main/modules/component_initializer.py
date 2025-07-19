"""
Component initialization module for the trading bot.
Handles initialization of all trading system components with proper error handling.
"""

import asyncio
import os
from typing import Dict, Optional, Callable
from datetime import datetime

# Updated imports to use the new package structure
from models.lstm_attention import AttentionLSTM
from models.temporal_fusion_transformer import TFTModel
from models.ensemble import EnsembleModel
from models.reinforcement_learning.multi_agent_system import MultiAgentTradingSystem

from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer

from trading.regime_detector import MarketRegimeDetector
from trading.strategies.momentum import MomentumStrategy
from trading.strategies.mean_reversion import MeanReversionStrategy
from trading.strategies.arbitrage import ArbitrageStrategy
from trading.strategies.market_making import MarketMakingStrategy
from trading.risk_manager import RiskManager
from trading.order_executor import OrderExecutor
from trading.adaptive_strategy_manager import AdaptiveStrategyManager
from trading.execution import AdvancedExecutor
from trading.dynamic_hedging import DynamicHedgingSystem
from trading.position_sizer import PositionSizer
from trading.optimization import HRPOptimizer, BlackLittermanOptimizer
from trading.optimization.black_litterman import CryptoViewGenerator
from trading.portfolio import PortfolioAnalytics, PortfolioMonitor, LogAlertHandler
from trading.portfolio.dashboard_runner import DashboardManager

from exchange.hyperliquid_client import HyperliquidClient

from utils.logger import setup_logger
from utils.database import DatabaseManager

logger = setup_logger(__name__)


class ComponentInitializer:
    """Handles initialization of all trading bot components"""
    
    def __init__(self, config: Dict, health_tracker=None):
        self.config = config
        self.health_tracker = health_tracker
        self.components = {}
        
    async def initialize_all_components(self) -> Dict:
        """Initialize all bot components with comprehensive error handling"""
        logger.info("Initializing trading bot components...")
        
        # Component initialization order matters - define dependencies
        init_order = [
            ('exchange', [], self._init_exchange),
            ('risk_manager', [], self._init_risk_management),
            ('portfolio', ['risk_manager'], self._init_portfolio),
            ('dashboard', ['portfolio', 'risk_manager'], self._init_dashboard),
            ('order_executor', ['exchange'], self._init_order_execution),
            ('data_collector', [], self._init_data_components),
            ('ml_models', [], self._init_ml_models),
            ('strategies', ['risk_manager'], self._init_strategies),
            ('hedging', ['risk_manager'], self._init_hedging),
            ('database', [], self._init_database)
        ]
        
        for component_name, dependencies, init_func in init_order:
            try:
                # Register component
                if self.health_tracker:
                    self.health_tracker.register_component(component_name, dependencies)
                    
                    # Check dependencies
                    if not self.health_tracker.check_dependencies(component_name):
                        failed_deps = [
                            dep for dep in dependencies 
                            if self.health_tracker.component_status.get(dep) != 'healthy'
                        ]
                        raise RuntimeError(f"Dependencies not ready: {failed_deps}")
                
                # Initialize component
                logger.info(f"Initializing {component_name}...")
                await init_func()
                
                # Mark as healthy
                if self.health_tracker:
                    self.health_tracker.update_status(component_name, 'healthy')
                logger.info(f"{component_name} initialized successfully")
                
            except Exception as e:
                if self.health_tracker:
                    self.health_tracker.update_status(component_name, 'failed', str(e))
                    
                    # Check cascade impact
                    affected = self.health_tracker.get_cascade_impact(component_name)
                    logger.error(f"Failed to initialize {component_name}, affecting: {affected}")
                
                # Determine if this is a critical failure
                critical_components = {'exchange', 'risk_manager', 'order_executor', 'data_collector'}
                if component_name in critical_components:
                    raise
                else:
                    logger.warning(f"Continuing without {component_name}")
        
        logger.info("All components initialized successfully")
        return self.components
    
    async def _init_exchange(self):
        """Initialize exchange client"""
        private_key = self.config.get('exchange.private_key')
        if not private_key:
            raise ValueError("Exchange private key not configured")
            
        exchange = HyperliquidClient(
            private_key=private_key,
            testnet=self.config.get('exchange.testnet', False)
        )
        
        # Test connection immediately
        try:
            test_result = await exchange.test_connection()
            if not test_result:
                raise ConnectionError("Exchange connection test failed")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to exchange: {e}")
            
        self.components['exchange'] = exchange
    
    async def _init_risk_management(self):
        """Initialize risk management components"""
        initial_capital = self.config.get('trading.initial_capital', 100000)
        if initial_capital <= 0:
            raise ValueError(f"Invalid initial capital: {initial_capital}")
            
        risk_manager = RiskManager(initial_capital=initial_capital)
        position_sizer = PositionSizer(risk_manager)
        
        # Apply custom risk parameters from config
        risk_params = self.config.get('risk', {})
        for param, value in risk_params.items():
            if hasattr(risk_manager.risk_params, param):
                risk_manager.risk_params[param] = value
                
        self.components['risk_manager'] = risk_manager
        self.components['position_sizer'] = position_sizer
    
    async def _init_portfolio(self):
        """Initialize portfolio components"""
        hrp_optimizer = HRPOptimizer()
        portfolio_analytics = PortfolioAnalytics()
        
        # Portfolio monitoring with error handling
        alert_handlers = []
        
        # Multiple alert handlers
        alert_configs = self.config.get('alerts', {})
        
        # Log alerts
        if alert_configs.get('log_alerts', True):
            try:
                alert_handlers.append(LogAlertHandler("logs/portfolio_alerts.log"))
            except Exception as e:
                logger.warning(f"Failed to create log alert handler: {e}")
        
        # Email alerts (if configured)
        if alert_configs.get('email_alerts', False):
            try:
                from trading.portfolio.email_alert_handler import EmailAlertHandler
                alert_handlers.append(EmailAlertHandler(alert_configs['email_config']))
            except Exception as e:
                logger.warning(f"Failed to create email alert handler: {e}")
        
        # Slack alerts (if configured)
        if alert_configs.get('slack_alerts', False):
            try:
                from trading.portfolio.slack_alert_handler import SlackAlertHandler
                alert_handlers.append(SlackAlertHandler(alert_configs['slack_config']))
            except Exception as e:
                logger.warning(f"Failed to create Slack alert handler: {e}")
                
        portfolio_monitor = PortfolioMonitor(
            alert_handlers=alert_handlers, 
            check_interval=self.config.get('monitoring.portfolio_check_interval', 300)
        )
        
        self.components['hrp_optimizer'] = hrp_optimizer
        self.components['portfolio_analytics'] = portfolio_analytics
        self.components['portfolio_monitor'] = portfolio_monitor
    
    async def _init_dashboard(self):
        """Initialize dashboard"""
        dashboard_port = self.config.get('dashboard.port', 5000)
        if self.config.get('dashboard.enabled', True):
            dashboard_manager = DashboardManager(
                portfolio_analytics=self.components['portfolio_analytics'],
                portfolio_monitor=self.components['portfolio_monitor'],
                risk_manager=self.components['risk_manager'],
                port=dashboard_port
            )
            self.components['dashboard_manager'] = dashboard_manager
        else:
            self.components['dashboard_manager'] = None
    
    async def _init_order_execution(self):
        """Initialize order execution components"""
        exchange = self.components['exchange']
        order_executor = OrderExecutor(exchange)
        advanced_executor = AdvancedExecutor(exchange)
        
        # Apply execution parameters from config
        exec_params = self.config.get('execution', {})
        if exec_params:
            order_executor.params.update(exec_params)
            advanced_executor.params.update(exec_params)
            
        self.components['order_executor'] = order_executor
        self.components['advanced_executor'] = advanced_executor
    
    async def _init_data_components(self):
        """Initialize data components"""
        redis_host = self.config.get('data.redis_host', 'localhost')
        redis_port = self.config.get('data.redis_port', 6379)
        
        data_collector = DataCollector(
            redis_host=redis_host,
            redis_port=redis_port
        )
        
        # Test Redis connection with retry
        for attempt in range(3):
            if data_collector.test_connection():
                break
            if attempt < 2:
                logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2)
        else:
            raise ConnectionError("Failed to connect to Redis after 3 attempts")
            
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        
        self.components['data_collector'] = data_collector
        self.components['preprocessor'] = preprocessor
        self.components['feature_engineer'] = feature_engineer
    
    async def _init_ml_models(self):
        """Initialize ML models"""
        import torch
        
        device = self.config.get('ml_models.device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate device availability
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        # Initialize models with error handling
        ensemble_predictor = EnsembleModel(device=device)
        regime_detector = MarketRegimeDetector()
        bl_optimizer = BlackLittermanOptimizer()
        view_generator = CryptoViewGenerator()
        
        # Initialize Multi-Agent RL System
        rl_system = MultiAgentTradingSystem(device=device)
        
        # Load pre-trained models with validation
        model_paths = self.config.get('ml_models.model_paths', {})
        
        # Load ensemble models
        if 'ensemble' in model_paths and os.path.exists(model_paths['ensemble']):
            try:
                ensemble_predictor.load_models(model_paths['ensemble'])
                logger.info("Loaded ensemble models")
            except Exception as e:
                logger.warning(f"Failed to load ensemble models: {e}")
        
        # Load RL agents
        rl_model_path = model_paths.get('rl_agents', 'models/rl_agents/')
        if os.path.exists(rl_model_path):
            try:
                rl_system.load_all_agents(rl_model_path)
                logger.info("Loaded RL agents")
            except Exception as e:
                logger.warning(f"Failed to load RL agents: {e}")
                
        self.components['ensemble_predictor'] = ensemble_predictor
        self.components['regime_detector'] = regime_detector
        self.components['bl_optimizer'] = bl_optimizer
        self.components['view_generator'] = view_generator
        self.components['rl_system'] = rl_system
    
    async def _init_strategies(self):
        """Initialize trading strategies"""
        risk_manager = self.components['risk_manager']
        
        adaptive_strategy_manager = AdaptiveStrategyManager(risk_manager)
        strategies = {}
        
        # Initialize each strategy with error handling
        strategy_configs = [
            ('momentum', MomentumStrategy),
            ('mean_reversion', MeanReversionStrategy),
            ('arbitrage', ArbitrageStrategy),
            ('market_making', MarketMakingStrategy)
        ]
        
        for name, StrategyClass in strategy_configs:
            if self.config.get(f'strategies.{name}.enabled', True):
                try:
                    # Pass strategy-specific config
                    strategy_config = self.config.get(f'strategies.{name}', {})
                    strategies[name] = StrategyClass(
                        risk_manager,
                        **strategy_config
                    )
                    logger.info(f"Initialized {name} strategy")
                except Exception as e:
                    logger.error(f"Failed to initialize {name} strategy: {e}")
                    strategies[name] = None
                    
        self.components['adaptive_strategy_manager'] = adaptive_strategy_manager
        self.components['strategies'] = strategies
    
    async def _init_hedging(self):
        """Initialize hedging system"""
        if self.config.get('hedging.enabled', True):
            hedging_system = DynamicHedgingSystem()
            self.components['hedging_system'] = hedging_system
        else:
            self.components['hedging_system'] = None
    
    async def _init_database(self):
        """Initialize database"""
        db = DatabaseManager()
        
        # Test with retry and connection pooling
        for attempt in range(3):
            if db.test_connection():
                break
            if attempt < 2:
                logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2)
        else:
            raise ConnectionError("Failed to connect to database after 3 attempts")
        
        # Initialize database schema if needed
        try:
            db.initialize_schema()
        except Exception as e:
            logger.warning(f"Failed to initialize database schema: {e}")
            
        self.components['db'] = db
    
    async def cleanup_partial_initialization(self):
        """Clean up partially initialized components"""
        logger.info("Cleaning up partial initialization...")
        
        # Get all components that need cleanup
        cleanup_order = [
            'dashboard_manager', 'portfolio_monitor', 'order_executor',
            'advanced_executor', 'data_collector', 'db', 'exchange'
        ]
        
        for component_name in cleanup_order:
            if component_name in self.components:
                try:
                    obj = self.components[component_name]
                    if hasattr(obj, 'close'):
                        if asyncio.iscoroutinefunction(obj.close):
                            await obj.close()
                        else:
                            obj.close()
                    elif hasattr(obj, 'cleanup'):
                        obj.cleanup()
                    elif hasattr(obj, 'shutdown'):
                        obj.shutdown()
                except Exception as e:
                    logger.warning(f"Failed to cleanup {component_name}: {e}")
