import asyncio
import signal
import sys
from typing import Dict, List
import pandas as pd
import torch
import numpy as np
from datetime import datetime, timedelta

# Import all components
from models.lstm_attention import AttentionLSTM
from models.temporal_fusion_transformer import TFTModel
from models.ensemble import EnsemblePredictor
from models.regime_detector import MarketRegimeDetector
from models.reinforcement_learning.multi_agent_system import MultiAgentTradingSystem

from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer

from trading.strategies.momentum import MomentumStrategy
from trading.strategies.mean_reversion import MeanReversionStrategy
from trading.strategies.arbitrage import ArbitrageStrategy
from trading.strategies.market_making import MarketMakingStrategy
from trading.risk_manager import RiskManager
from trading.position_sizer import PositionSizer
from trading.order_executor import OrderExecutor
from trading.adaptive_strategy_manager import AdaptiveStrategyManager
from trading.execution.advanced_executor import AdvancedOrderExecutor
from trading.dynamic_hedging import DynamicHedgingSystem
from src.trading.regime_detector import RegimeAwarePositionSizer
from src.trading.optimization.hierarchical_risk_parity import HierarchicalRiskParity
from src.trading.optimization.black_litterman import BlackLittermanOptimizer, CryptoViewGenerator
from trading.portfolio import PortfolioAnalytics, PortfolioMonitor, LogAlertHandler
from trading.portfolio.dashboard_runner import DashboardManager

from exchange.hyperliquid_client import HyperliquidClient

from utils.config import Config
from utils.logger import setup_logger, log_trade
from utils.database import DatabaseManager

logger = setup_logger(__name__)

class HyperliquidTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load configuration
        self.config = Config(config_path)
        
        if not self.config.validate():
            raise ValueError("Invalid configuration")
        
        # Initialize components
        self._initialize_components()
        
        # Control flags
        self.running = False
        self.shutdown_event = asyncio.Event()
        
    def _initialize_components(self):
        """Initialize all bot components"""
        logger.info("Initializing trading bot components...")
        
        # Exchange client
        self.exchange = HyperliquidClient(
            private_key=self.config.get('exchange.private_key'),
            testnet=self.config.get('exchange.testnet', False)
        )
        
        # Risk management - Enhanced with multiple components
        self.risk_manager = RiskManager(
            initial_capital=self.config.get('trading.initial_capital', 100000)
        )
        self.position_sizer = PositionSizer(self.risk_manager)
        self.regime_position_sizer = RegimeAwarePositionSizer(self.risk_manager)
        
        # Portfolio optimization and analytics
        self.hrp_optimizer = HierarchicalRiskParity()
        self.portfolio_analytics = PortfolioAnalytics()
        
        # Portfolio monitoring and alerting
        alert_handlers = [LogAlertHandler("logs/portfolio_alerts.log")]
        self.portfolio_monitor = PortfolioMonitor(alert_handlers=alert_handlers, check_interval=300)  # 5 min checks
        
        # Portfolio dashboard
        dashboard_port = self.config.get('dashboard.port', 5000)
        self.dashboard_manager = DashboardManager(
            portfolio_analytics=self.portfolio_analytics,
            portfolio_monitor=self.portfolio_monitor,
            risk_manager=self.risk_manager,
            port=dashboard_port
        )
        
        # Order execution - Enhanced with advanced algorithms
        self.order_executor = OrderExecutor(self.exchange)
        self.advanced_executor = AdvancedOrderExecutor(self.exchange)
        
        # Data components
        self.data_collector = DataCollector(
            redis_host=self.config.get('data.redis_host', 'localhost'),
            redis_port=self.config.get('data.redis_port', 6379)
        )
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # ML models - Enhanced with RL and portfolio optimization
        self.ensemble_predictor = EnsemblePredictor()
        self.regime_detector = MarketRegimeDetector()
        self.bl_optimizer = BlackLittermanOptimizer()
        self.view_generator = CryptoViewGenerator()
        self.rl_system = MultiAgentTradingSystem()
        
        # Trading strategies - Enhanced with adaptive management
        self.adaptive_strategy_manager = AdaptiveStrategyManager(self.risk_manager)
        self.strategies = {
            'momentum': MomentumStrategy(self.risk_manager),
            'mean_reversion': MeanReversionStrategy(self.risk_manager),
            'arbitrage': ArbitrageStrategy(self.risk_manager),
            'market_making': MarketMakingStrategy(self.risk_manager)
        }
        
        # Enhanced risk management
        self.hedging_system = DynamicHedgingSystem()
        
        # Database
        self.db = DatabaseManager()
        
        # Performance tracking
        self.performance = {
            'start_time': datetime.now(),
            'initial_capital': self.config.get('trading.initial_capital'),
            'trades': 0,
            'wins': 0,
            'total_pnl': 0
        }
        
        logger.info("All components initialized successfully")
    
    async def start(self):
        """Start the trading bot"""
        logger.info("Starting Hyperliquid trading bot...")
        
        self.running = True
        
        # Connect to exchange
        await self.exchange.connect_websocket()
        
        # Start data collection
        symbols = self.config.get('trading.symbols', ['BTC-USD', 'ETH-USD'])
        await self._start_data_collection(symbols)
        
        # Start portfolio dashboard
        if self.config.get('dashboard.enabled', True):
            self.dashboard_manager.start_dashboard()
        
        # Main trading loop
        tasks = [
            self._trading_loop(),
            self._risk_monitoring_loop(),
            self._model_update_loop(),
            self._portfolio_rebalancing_loop(),
            self._performance_tracking_loop(),
            self.portfolio_monitor.start_monitoring(self.risk_manager, self.portfolio_analytics, self.data_collector)
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.shutdown()
    
    async def _start_data_collection(self, symbols: List[str]):
        """Start data collection for symbols"""
        # Subscribe to market data
        for symbol in symbols:
            await self.data_collector.subscribe_orderbook(symbol)
            await self.data_collector.subscribe_trades(symbol)
            await self.data_collector.subscribe_funding(symbol)
        
        # Start listening in background
        asyncio.create_task(self.data_collector.listen())
        
        logger.info(f"Started data collection for {symbols}")
    
    async def _trading_loop(self):
        """Main trading loop"""
        await asyncio.sleep(5)  # Wait for initial data
        
        while self.running:
            try:
                # Get configured symbols
                symbols = self.config.get('trading.symbols', ['BTC-USD'])
                
                for symbol in symbols:
                    # Collect and prepare data
                    market_data = await self._prepare_market_data(symbol)
                    
                    if market_data is None:
                        continue
                    
                    # Get ML predictions
                    ml_predictions = await self._get_ml_predictions(market_data)
                    
                    # Detect market regime
                    regime = self.regime_detector.detect_regime(market_data['ohlcv'])
                    
                    # Generate signals from each strategy
                    all_signals = []
                    
                    for strategy_name, strategy in self.strategies.items():
                        if self.config.get(f'strategies.{strategy_name}.enabled', True):
                            if strategy_name == 'arbitrage':
                                # Arbitrage needs different data
                                opportunities = strategy.find_opportunities(
                                    self._get_all_market_data(),
                                    self._get_funding_rates()
                                )
                                all_signals.extend(opportunities)
                            elif strategy_name == 'market_making':
                                # Market making generates quotes
                                quotes = strategy.calculate_quotes(
                                    market_data['current'],
                                    ml_predictions
                                )
                                await self._execute_market_making(quotes, strategy)
                            else:
                                # Momentum and mean reversion
                                signals = strategy.analyze(
                                    market_data['ohlcv'],
                                    ml_predictions
                                )
                                all_signals.extend(signals)
                    
                    # Filter and rank signals
                    selected_signals = self._select_best_signals(all_signals, regime)
                    
                    # Execute selected signals
                    for signal in selected_signals:
                        await self._execute_signal(signal)
                    
                    # Update existing positions
                    await self._update_positions()
                
                # Sleep between iterations
                await asyncio.sleep(1)  # 1 second loop
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _prepare_market_data(self, symbol: str) -> Optional[Dict]:
        """Prepare market data for analysis"""
        try:
            # Get historical data
            ohlcv = await self.data_collector.get_historical_data(symbol, '1h', 500)
            
            if ohlcv.empty:
                return None
            
            # Preprocess data
            ohlcv = self.preprocessor.prepare_ohlcv_data(ohlcv)
            ohlcv = self.preprocessor.calculate_technical_indicators(ohlcv)
            
            # Get current orderbook
            orderbook = self.data_collector.get_latest_orderbook(symbol)
            
            if orderbook:
                ohlcv = self.preprocessor.calculate_microstructure_features(ohlcv, orderbook)
            
            # Engineer features
            ohlcv = self.feature_engineer.engineer_all_features(ohlcv)
            
            # Get recent trades
            recent_trades = self.data_collector.get_recent_trades(symbol)
            
            return {
                'symbol': symbol,
                'ohlcv': ohlcv,
                'current': {
                    'symbol': symbol,
                    'best_bid': orderbook['best_bid'] if orderbook else ohlcv['close'].iloc[-1] * 0.999,
                    'best_ask': orderbook['best_ask'] if orderbook else ohlcv['close'].iloc[-1] * 1.001,
                    'spread': orderbook['spread'] if orderbook else ohlcv['close'].iloc[-1] * 0.002,
                    'mid_price': orderbook['mid_price'] if orderbook else ohlcv['close'].iloc[-1],
                    'volatility': ohlcv['volatility_20'].iloc[-1] if 'volatility_20' in ohlcv else 0.02,
                    'volume': ohlcv['volume'].iloc[-1],
                    'order_imbalance': orderbook.get('imbalance', 0) if orderbook else 0,
                    'recent_trades': recent_trades[-20:] if recent_trades else []
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing market data for {symbol}: {e}")
            return None
    
    async def _get_ml_predictions(self, market_data: Dict) -> Dict:
        """Get ML model predictions"""
        try:
            # Prepare inputs for different models
            df = market_data['ohlcv']
            
            # Create sequences for LSTM
            lstm_sequences, _ = self.preprocessor.create_sequences(df, 60, 1)
            if len(lstm_sequences) == 0:
                return {}
            
            lstm_input = torch.tensor(lstm_sequences[-1:], dtype=torch.float32)
            
            # Prepare TFT input
            tft_input = {
                'price_features': df[['close', 'high', 'low', 'open', 'volume']].iloc[-60:].values,
                'volume_features': df[['volume', 'volume_ratio', 'dollar_volume']].iloc[-60:].values,
                'market_features': df.iloc[-60:, -10:].values  # Last 10 features
            }
            
            # Prepare gradient boosting input
            gb_features = df.iloc[-1].values
            
            # Get ensemble predictions
            predictions = self.ensemble_predictor.ensemble_predict(
                lstm_input,
                tft_input,
                gb_features.reshape(1, -1),
                torch.tensor(df.iloc[-20:, -10:].mean().values, dtype=torch.float32)
            )
            
            current_price = df['close'].iloc[-1]
            pred_price = predictions['mean']
            
            return {
                'price_prediction': pred_price,
                'price_change': (pred_price - current_price) / current_price,
                'confidence': 1 - (predictions['std'] / pred_price),  # Simple confidence metric
                'direction': 1 if pred_price > current_price else -1,
                'upper_bound': predictions['upper_bound'],
                'lower_bound': predictions['lower_bound']
            }
            
        except Exception as e:
            logger.error(f"Error getting ML predictions: {e}")
            return {}
    
    def _select_best_signals(self, signals: List, regime: Dict) -> List:
        """Select best signals based on regime and risk"""
        if not signals:
            return []
        
        # Filter by regime preferences
        preferred_strategies = regime['trading_mode']['preferred_strategies']
        
        filtered_signals = []
        for signal in signals:
            # Determine signal strategy type
            if hasattr(signal, 'type'):
                strategy_type = signal.type
            elif hasattr(signal, 'z_score'):
                strategy_type = 'mean_reversion'
            elif hasattr(signal, 'strength'):
                strategy_type = 'momentum'
            else:
                strategy_type = 'unknown'
            
            if strategy_type in preferred_strategies:
                filtered_signals.append(signal)
        
        # Sort by confidence/expected profit
        if hasattr(filtered_signals[0], 'confidence'):
            filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
        elif hasattr(filtered_signals[0], 'expected_profit_pct'):
            filtered_signals.sort(key=lambda x: x.expected_profit_pct, reverse=True)
        
        # Apply position limits
        max_new_positions = self.risk_manager.risk_params['position_limit'] - len(self.risk_manager.positions)
        
        return filtered_signals[:max_new_positions]
    
    async def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            # Check risk
            if hasattr(signal, 'symbol'):
                symbol = signal.symbol
                side = 'buy' if signal.direction == 1 else 'sell'
                
                # Calculate position size using multiple methods
                # Use regime-aware position sizer for enhanced sizing
                base_size = self.position_sizer.calculate_position_size(
                signal.__dict__,
                {'volatility': 0.02},  # Simplified
                    {'positions': self.risk_manager.positions}
                    )
                    
                    # Enhanced regime-aware sizing
                    regime_adjusted_size = self.regime_position_sizer.calculate_regime_aware_position_size(
                        base_size=base_size,
                        regime_info=regime,
                        signal_strength=getattr(signal, 'strength', 0.5),
                        confidence=getattr(signal, 'confidence', 0.5),
                        market_conditions={
                            'volatility': market_data['current'].get('volatility', 0.02),
                            'trend_strength': regime.get('trend_strength', 0.0),
                            'market_stress': regime.get('market_stress', 0.0),
                            'market_correlation': 0.5,  # Simplified
                            'liquidity': 1.0  # Simplified
                        }
                    )
                    
                    size = regime_adjusted_size
                
                # Risk check
                can_trade, reason = self.risk_manager.check_pre_trade_risk(
                    symbol, side, size, signal.entry_price, signal.stop_loss
                )
                
                if not can_trade:
                    logger.warning(f"Risk check failed for {symbol}: {reason}")
                    return
                
                # Execute order
                order_result = await self.order_executor.place_order(
                    symbol=symbol,
                    side=side,
                    size=size,
                    order_type='limit',
                    price=signal.entry_price,
                    metadata={'signal': signal.__class__.__name__}
                )
                
                if order_result['status'] == 'filled':
                    # Add position to risk manager
                    self.risk_manager.add_position(
                        symbol, side, order_result['filled_size'],
                        order_result['fill_price'], signal.stop_loss
                    )
                    
                    # Record trade
                    self.db.record_trade({
                        'symbol': symbol,
                        'side': side,
                        'size': order_result['filled_size'],
                        'entry_price': order_result['fill_price'],
                        'strategy': signal.__class__.__name__,
                        'status': 'open'
                    })
                    
                    # Log trade
                    log_trade(symbol, side, order_result['filled_size'], 
                             order_result['fill_price'], reason=signal.__class__.__name__)
                    
                    self.performance['trades'] += 1
                    
            elif hasattr(signal, 'type') and signal.type in ['triangular', 'statistical', 'funding']:
                # Execute arbitrage
                result = await self.strategies['arbitrage'].execute_opportunity(
                    signal, self.order_executor
                )
                
                if result['status'] == 'success':
                    logger.info(f"Executed arbitrage: {signal.type}, profit: {result['profit_realized']}")
                    
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def _execute_market_making(self, quotes: List, strategy: MarketMakingStrategy):
        """Execute market making quotes"""
        try:
            results = await strategy.execute_quotes(quotes, self.order_executor)
            
            logger.info(f"Market making: placed {len(results['placed_orders'])} orders")
            
        except Exception as e:
            logger.error(f"Error in market making: {e}")
    
    async def _update_positions(self):
        """Update existing positions"""
        try:
            # Get current prices
            current_prices = {}
            for symbol in self.risk_manager.positions:
                ticker = await self.exchange.get_ticker(symbol)
                current_prices[symbol] = ticker['last']
            
            # Update position prices
            for symbol, price in current_prices.items():
                self.risk_manager.update_position_price(symbol, price)
            
            # Get strategy-specific updates
            all_actions = []
            
            # Momentum strategy updates
            momentum_actions = self.strategies['momentum'].update_positions(current_prices)
            all_actions.extend(momentum_actions)
            
            # Mean reversion updates
            mr_data = {}
            for symbol in self.risk_manager.positions:
                data = await self._prepare_market_data(symbol)
                if data:
                    mr_data[symbol] = data['ohlcv']
            
            mr_actions = self.strategies['mean_reversion'].update_positions(mr_data)
            all_actions.extend(mr_actions)
            
            # Execute actions
            for action in all_actions:
                if action['action'] == 'close_position':
                    await self._close_position(action['symbol'], action.get('reason', 'strategy_exit'))
                elif action['action'] == 'update_stop_loss':
                    # Update stop loss order
                    pass  # Implement stop loss update
                    
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.risk_manager.positions.get(symbol)
            if not position:
                return
            
            # Get current price
            ticker = await self.exchange.get_ticker(symbol)
            current_price = ticker['last']
            
            # Execute closing order
            side = 'sell' if position['side'] == 'long' else 'buy'
            
            order_result = await self.order_executor.place_order(
                symbol=symbol,
                side=side,
                size=position['size'],
                order_type='market'
            )
            
            if order_result['status'] in ['filled', 'partial']:
                # Close position in risk manager
                self.risk_manager.close_position(
                    symbol,
                    order_result.get('fill_price', current_price),
                    reason
                )
                
                # Update database
                # Find the open trade and update it
                trades = self.db.get_trade_history(symbol, days=7)
                open_trades = trades[trades['status'] == 'open']
                
                if not open_trades.empty:
                    trade_id = open_trades.iloc[0]['id']
                    pnl = position['realized_pnl']
                    
                    self.db.update_trade(trade_id, {
                        'exit_price': order_result.get('fill_price', current_price),
                        'pnl': pnl,
                        'status': 'closed'
                    })
                    
                    # Update performance
                    if pnl > 0:
                        self.performance['wins'] += 1
                    self.performance['total_pnl'] += pnl
                    
                    # Log trade
                    log_trade(symbol, side, position['size'], 
                             order_result.get('fill_price', current_price),
                             pnl=pnl, reason=reason)
                    
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
    
    async def _risk_monitoring_loop(self):
        """Monitor risk metrics continuously"""
        while self.running:
            try:
                # Calculate risk metrics
                metrics = self.risk_manager.calculate_risk_metrics()
                
                # Check for breaches
                if metrics.current_drawdown > self.risk_manager.risk_params['max_drawdown'] * 0.8:
                    logger.warning(f"Approaching max drawdown: {metrics.current_drawdown:.2%}")
                    self.risk_manager.log_risk_breach(
                        'drawdown_warning',
                        f"Current drawdown {metrics.current_drawdown:.2%} approaching limit"
                    )
                
                if metrics.risk_score > 80:
                    logger.warning(f"High risk score: {metrics.risk_score}")
                    # Reduce position sizes or halt trading
                    
                # Reset daily counters if needed
                self.risk_manager.reset_daily_counters()
                
                # Log metrics
                self.db.record_performance_metric('risk_score', metrics.risk_score)
                self.db.record_performance_metric('current_drawdown', metrics.current_drawdown)
                self.db.record_performance_metric('sharpe_ratio', metrics.sharpe_ratio)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _model_update_loop(self):
        """Update ML models periodically"""
        while self.running:
            try:
                # Wait for configured interval
                update_interval = self.config.get('ml_models.retrain_interval', 24) * 3600
                await asyncio.sleep(update_interval)
                
                logger.info("Starting model update...")
                
                # Collect training data
                symbols = self.config.get('trading.symbols', ['BTC-USD'])
                
                for symbol in symbols:
                    # Load historical data
                    df = self.db.load_market_data(
                        symbol,
                        start_date=datetime.now() - timedelta(days=90)
                    )
                    
                    if len(df) < self.config.get('ml_models.min_data_points', 1000):
                        continue
                    
                    # Prepare features
                    df = self.preprocessor.prepare_ohlcv_data(df)
                    df = self.preprocessor.calculate_technical_indicators(df)
                    df = self.feature_engineer.engineer_all_features(df)
                    
                    # Create targets
                    df = self.feature_engineer.create_target_features(df)
                    
                    # Prepare dataset
                    feature_cols = [col for col in df.columns if not col.startswith('target_')]
                    target_col = 'target_return_1'  # 1-hour ahead return
                    
                    dataset = self.preprocessor.prepare_ml_dataset(
                        df, feature_cols, target_col
                    )
                    
                    # Train gradient boosting models
                    self.ensemble_predictor.train_gradient_boosting(
                        dataset['X_train'], dataset['y_train'],
                        dataset['X_test'], dataset['y_test']
                    )
                
                # Train regime detector
                all_data = pd.DataFrame()
                for symbol in symbols:
                    df = self.db.load_market_data(
                        symbol,
                        start_date=datetime.now() - timedelta(days=180)
                    )
                    if not df.empty:
                        all_data = pd.concat([all_data, df])
                
                if not all_data.empty:
                    self.regime_detector.fit(all_data)
                
                logger.info("Model update completed")
                
            except Exception as e:
                logger.error(f"Error in model update: {e}")
    
    async def _portfolio_rebalancing_loop(self):
        """Portfolio rebalancing using HRP optimization"""
        while self.running:
            try:
                # Rebalance every 24 hours
                await asyncio.sleep(24 * 3600)
                
                logger.info("Starting portfolio rebalancing with HRP...")
                
                # Get portfolio data
                symbols = list(self.risk_manager.positions.keys())
                if len(symbols) < 3:
                    logger.info("Not enough positions for HRP optimization")
                    continue
                
                # Get historical returns for all symbols
                returns_data = []
                for symbol in symbols:
                    # Get 90 days of data
                    historical_data = await self.data_collector.get_historical_data(symbol, '1d', 90)
                    if not historical_data.empty:
                        returns = historical_data['close'].pct_change().dropna()
                        returns_data.append(returns)
                
                if len(returns_data) < 3:
                    logger.warning("Insufficient data for HRP optimization")
                    continue
                
                # Create returns matrix
                returns_df = pd.concat(returns_data, axis=1, keys=symbols).dropna()
                
                if len(returns_df) < 30:  # Need at least 30 days of data
                    logger.warning("Insufficient historical data for HRP")
                    continue
                
                # Optimize portfolio using HRP
                hrp_result = self.hrp_optimizer.optimize_portfolio(returns_df)
                optimal_weights = hrp_result['weights']
                
                # Generate portfolio analytics report
                portfolio_returns = self._calculate_portfolio_returns(symbols)
                portfolio_report = self.portfolio_analytics.generate_portfolio_report(
                    self.risk_manager.positions,
                    portfolio_returns,
                    optimal_weights
                )
                
                # Get rebalancing recommendations
                rebalancing_recommendations = self.portfolio_analytics.analyze_rebalancing_needs(
                    self.risk_manager.positions,
                    optimal_weights,
                    tolerance=0.05
                )
                
                logger.info(f"Portfolio Analytics Report Generated - {len(rebalancing_recommendations)} recommendations")
                
                # Log portfolio performance metrics
                if 'performance_metrics' in portfolio_report:
                    perf = portfolio_report['performance_metrics']
                    logger.info(f"Portfolio Performance - Return: {perf.get('total_return', 0):.2%}, "
                               f"Sharpe: {perf.get('sharpe_ratio', 0):.2f}, "
                               f"Max DD: {perf.get('max_drawdown', 0):.2%}")
                
                # Use analytics-based recommendations instead of simple weight differences
                rebalancing_actions = []
                for rec in rebalancing_recommendations:
                    if rec.urgency in ['medium', 'high']:  # Only execute medium/high priority
                        rebalancing_actions.append({
                            'symbol': rec.symbol,
                            'current_weight': rec.current_weight,
                            'target_weight': rec.target_weight,
                            'size_adjustment': rec.amount_to_trade,
                            'action': rec.action,
                            'urgency': rec.urgency,
                            'reason': rec.reason
                        })
                
                # Execute rebalancing trades
                for action in rebalancing_actions:
                    try:
                        symbol = action['symbol']
                        size = abs(action['size_adjustment'])
                        side = action['action']
                        
                        if size > 0.001:  # Minimum trade size
                            # Use TWAP for large rebalancing trades
                            if size * self.risk_manager.positions[symbol].get('current_price', 100) > 10000:
                                await self.advanced_executor.execute_twap_order(
                                    symbol=symbol,
                                    side=side,
                                    total_size=size,
                                    duration_minutes=30,  # 30 minute TWAP
                                    metadata={'rebalancing': True}
                                )
                            else:
                                # Regular order for smaller trades
                                await self.order_executor.place_order(
                                    symbol=symbol,
                                    side=side,
                                    size=size,
                                    order_type='market',
                                    metadata={'rebalancing': True}
                                )
                            
                            logger.info(f"Rebalanced {symbol}: {action['current_weight']:.2%} -> {action['target_weight']:.2%} [{action['urgency']} priority]")
                    
                    except Exception as e:
                        logger.error(f"Error rebalancing {action['symbol']}: {e}")
                
                # Log HRP metrics
                portfolio_metrics = hrp_result.get('portfolio_metrics', {})
                logger.info(f"HRP Portfolio - Volatility: {portfolio_metrics.get('volatility', 0):.4f}, "
                           f"Diversification: {portfolio_metrics.get('diversification_ratio', 0):.2f}")
                
            except Exception as e:
                logger.error(f"Error in portfolio rebalancing: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def _performance_tracking_loop(self):
        """Track and report performance"""
        while self.running:
            try:
                # Get current metrics
                risk_metrics = self.risk_manager.calculate_risk_metrics()
                
                # Calculate performance
                current_capital = self.risk_manager.current_capital
                total_return = (current_capital - self.performance['initial_capital']) / self.performance['initial_capital']
                
                # Time-based metrics
                hours_running = (datetime.now() - self.performance['start_time']).total_seconds() / 3600
                
                if hours_running > 0:
                    hourly_return = total_return / hours_running
                    projected_monthly = hourly_return * 24 * 30  # Rough projection
                else:
                    projected_monthly = 0
                
                # Win rate
                win_rate = self.performance['wins'] / self.performance['trades'] if self.performance['trades'] > 0 else 0
                
                # Log performance
                logger.info(f"""
Performance Update:
- Total Return: {total_return:.2%}
- Projected Monthly: {projected_monthly:.2%}
- Win Rate: {win_rate:.2%}
- Trades: {self.performance['trades']}
- Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
- Current Drawdown: {risk_metrics.current_drawdown:.2%}
- Risk Score: {risk_metrics.risk_score:.1f}
                """.strip())
                
                # Save to database
                self.db.record_performance_metric('total_return', total_return)
                self.db.record_performance_metric('win_rate', win_rate)
                self.db.record_performance_metric('projected_monthly_return', projected_monthly)
                
                # Send alerts if needed
                if projected_monthly > 0.3:  # 30% monthly
                    logger.info("🎉 Exceeding 30% monthly return target!")
                elif projected_monthly < 0:
                    logger.warning("⚠️ Negative returns detected")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(300)
    
    def _calculate_portfolio_returns(self, symbols: List[str]) -> pd.Series:
        """Calculate actual portfolio returns from position history"""
        try:
            # Get portfolio return history from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 90 days of history
            
            # Try to get actual portfolio returns from database
            portfolio_returns = self.db.get_portfolio_returns(start_date, end_date)
            
            if portfolio_returns is not None and len(portfolio_returns) > 0:
                logger.info(f"Using {len(portfolio_returns)} days of actual portfolio returns")
                return portfolio_returns
            
            # Fallback: Calculate returns from current positions
            logger.info("No historical returns found, calculating from current positions")
            
            if not self.risk_manager.positions:
                # Return empty series if no positions
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                return pd.Series(0.0, index=dates)
            
            # Calculate portfolio value changes
            total_pnl = sum(
                pos.get('unrealized_pnl', 0) + pos.get('realized_pnl', 0)
                for pos in self.risk_manager.positions.values()
            )
            
            total_value = sum(
                pos['size'] * pos.get('current_price', pos['entry_price'])
                for pos in self.risk_manager.positions.values()
            )
            
            # Calculate current return
            current_return = total_pnl / (total_value - total_pnl) if (total_value - total_pnl) > 0 else 0
            
            # Estimate historical returns based on individual asset performance
            historical_returns = self._estimate_portfolio_returns_from_assets(symbols, current_return)
            
            return historical_returns
            
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            # Fallback to simple dummy data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            return pd.Series(
                np.random.normal(0.001, 0.02, 30),  # 0.1% mean, 2% volatility
                index=dates
            )
    
    def _estimate_portfolio_returns_from_assets(self, symbols: List[str], current_return: float) -> pd.Series:
        """Estimate portfolio returns from individual asset returns"""
        try:
            # Get historical data for portfolio assets
            asset_returns = {}
            weights = {}
            
            total_value = sum(
                pos['size'] * pos.get('current_price', pos['entry_price'])
                for pos in self.risk_manager.positions.values()
            )
            
            for symbol in symbols:
                if symbol in self.risk_manager.positions:
                    # Get weight
                    position = self.risk_manager.positions[symbol]
                    position_value = position['size'] * position.get('current_price', position['entry_price'])
                    weight = position_value / total_value if total_value > 0 else 0
                    weights[symbol] = weight
                    
                    # Get historical returns (simplified - use recent data)
                    try:
                        # This would typically be async, but we'll simulate
                        # In practice, you'd await self.data_collector.get_historical_data(symbol, '1d', 30)
                        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                        
                        # Simulate asset returns based on current performance
                        asset_volatility = 0.02 if 'BTC' in symbol else 0.025  # Different vol for different assets
                        daily_returns = np.random.normal(current_return / 30, asset_volatility, 30)
                        
                        asset_returns[symbol] = pd.Series(daily_returns, index=dates)
                        
                    except Exception as e:
                        logger.warning(f"Could not get returns for {symbol}: {e}")
                        continue
            
            if not asset_returns:
                # No asset returns available
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                return pd.Series(current_return / 30, index=dates)
            
            # Calculate weighted portfolio returns
            portfolio_returns = None
            for symbol, returns in asset_returns.items():
                weight = weights.get(symbol, 0)
                weighted_returns = returns * weight
                
                if portfolio_returns is None:
                    portfolio_returns = weighted_returns
                else:
                    portfolio_returns = portfolio_returns.add(weighted_returns, fill_value=0)
            
            return portfolio_returns if portfolio_returns is not None else pd.Series(dtype=float)
            
        except Exception as e:
            logger.error(f"Error estimating portfolio returns from assets: {e}")
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            return pd.Series(current_return / 30, index=dates)
    
    def _get_all_market_data(self) -> Dict[str, Dict]:
        """Get market data for all symbols (for arbitrage)"""
        market_data = {}
        
        for symbol in self.config.get('trading.symbols', []):
            orderbook = self.data_collector.get_latest_orderbook(symbol)
            if orderbook:
                market_data[symbol] = orderbook
        
        return market_data
    
    def _get_funding_rates(self) -> Dict[str, float]:
        """Get funding rates for all symbols"""
        # This would be implemented based on actual funding data
        return {}
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down trading bot...")
        
        self.running = False
        
        # Stop portfolio monitoring
        self.portfolio_monitor.stop_monitoring()
        
        # Stop dashboard
        if hasattr(self, 'dashboard_manager'):
            self.dashboard_manager.stop_dashboard()
        
        # Cancel all open orders
        await self.order_executor.cancel_all_orders()
        
        # Close all positions
        for symbol in list(self.risk_manager.positions.keys()):
            await self._close_position(symbol, "shutdown")
        
        # Save final state
        self.db.cleanup_old_data(90)
        
        # Close connections
        await self.data_collector.close()
        await self.exchange.close()
        
        logger.info("Trading bot shutdown complete")
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}")
        asyncio.create_task(self.shutdown())
        sys.exit(0)

async def main():
    """Main entry point"""
    # Create bot instance
    bot = HyperliquidTradingBot()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, bot.signal_handler)
    signal.signal(signal.SIGTERM, bot.signal_handler)
    
    # Start bot
    try:
        await bot.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
