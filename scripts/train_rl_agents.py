"""
RL Training Script - Multi-agent reinforcement learning system trainer
Trains the multi-agent RL system using historical data with risk management
integration for safe position sizing and realistic trading constraints.

File: train_rl_agents.py
Modified: 2025-07-19
"""

import os
import sys
import argparse
import asyncio
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reinforcement_learning.multi_agent_system import (
    MultiAgentTradingSystem, 
    CryptoTradingEnvironment
)
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from trading.risk_manager import RiskManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


DATA_DIR = '../src/models/training_data'
PARENT_DIR = '/historical/real'
# PARENT_DIR = '/historical/sample'

class RiskAwareTradingEnvironment(CryptoTradingEnvironment):
    """
    Enhanced trading environment that integrates with risk management system
    to modify RL actions based on risk constraints.
    """
    
    def __init__(
        self, 
        market_data: pd.DataFrame, 
        initial_balance: float = 100000,
        risk_manager: Optional[RiskManager] = None
    ):
        super().__init__(market_data, initial_balance)
        self.risk_manager = risk_manager or RiskManager(initial_balance)
        
    def _execute_action(self, action: int) -> float:
        """
        Execute trading action with risk management constraints.
        
        This overrides the parent method to apply risk management rules
        before executing trades.
        """
        current_price = self.market_data['close'].iloc[self.current_step]
        original_action = action
        
        # Get original trade size from action
        if action == 0:  # Hold
            trade_size = 0
        elif action == 1:  # Buy 25%
            trade_size = self.balance * 0.25 / current_price
        elif action == 2:  # Buy 50%
            trade_size = self.balance * 0.50 / current_price
        elif action == 3:  # Buy 100%
            trade_size = self.balance / current_price
        elif action == 4:  # Sell 25%
            trade_size = -self.position * 0.25
        elif action == 5:  # Sell 50%
            trade_size = -self.position * 0.50
        else:  # Sell 100%
            trade_size = -self.position
        
        # Apply risk management constraints
        if trade_size != 0:
            symbol = self.market_data.attrs.get('symbol', 'BTC-USD')
            side = 'buy' if trade_size > 0 else 'sell'
            abs_size = abs(trade_size)
            
            # Calculate stop loss based on volatility
            volatility = self.market_data.get('volatility_20', pd.Series([0.02])).iloc[self.current_step]
            stop_loss = current_price * (1 - 2 * volatility) if side == 'buy' else current_price * (1 + 2 * volatility)
            
            # Check with risk manager
            can_trade, reason = self.risk_manager.check_pre_trade_risk(
                symbol, side, abs_size, current_price, stop_loss
            )
            
            if not can_trade:
                # Risk check failed - modify action
                if "position_limit" in reason:
                    # Reduce to smaller position
                    if action in [3, 6]:  # 100% actions
                        action = action - 1  # Reduce to 50%
                    elif action in [2, 5]:  # 50% actions
                        action = action - 1  # Reduce to 25%
                    else:
                        action = 0  # Hold
                elif "max_position_size" in reason:
                    # Calculate maximum allowed size
                    max_allowed = self.risk_manager.calculate_max_position_size(
                        self.balance, current_price, volatility
                    )
                    if max_allowed < abs_size * 0.25:
                        action = 0  # Can't even do 25%, so hold
                    elif max_allowed < abs_size * 0.5:
                        action = 1 if side == 'buy' else 4  # Only 25%
                    elif max_allowed < abs_size:
                        action = 2 if side == 'buy' else 5  # Only 50%
                else:
                    # Other risk constraint - hold
                    action = 0
                
                # Log the action modification
                logger.debug(f"Risk management modified action from {original_action} to {action}: {reason}")
        
        # Execute the potentially modified action
        reward = super()._execute_action(action)
        
        # Additional reward adjustment based on risk compliance
        if original_action != action and action == 0:
            # Penalize for being blocked by risk management
            reward -= 1
        elif original_action != action:
            # Small penalty for size reduction
            reward -= 0.5
        
        return reward


class RLTrainingManager:
    """Manages the training of RL agents with proper data handling."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager(
            initial_capital=self.config.get('trading.initial_capital', 100000)
        )
        
        # Initialize multi-agent system
        device = self.config.get('ml_models.device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.rl_system = MultiAgentTradingSystem(device=device)
        
        # Data paths
        self.data_dir = f"{DATA_DIR}{PARENT_DIR}"
        self.models_dir = self.config.get('ml_models.rl_models_path', './models/rl_agents/')
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            # Return default config
            return {
                'environment': 'development',
                'trading': {
                    'initial_capital': 100000,
                    'symbols': ['BTC-USD', 'ETH-USD']
                },
                'ml_models': {
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'rl_models_path': './models/rl_agents/',
                    'rl_training_episodes': 1000
                }
            }
    
    def load_training_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load training data based on environment configuration."""
        environment = self.config.get('environment', 'development')
        
        if environment == 'development':
            return self._load_development_data(symbol)
        else:
            return self._load_live_data(symbol)
    
    def _load_development_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from development environment (sample data)."""
        try:
            # Map symbol to filename format used by download_sample_data.py
            symbol_map = {
                'BTC-USD': 'bitcoin',
                'ETH-USD': 'ethereum',
                'SOL-USD': 'solana'
            }
            
            base_name = symbol_map.get(symbol, symbol.lower().replace('-usd', ''))
            
            # Try different file formats
            file_paths = [
                os.path.join(self.data_dir, f"{base_name}_sample_data.csv"),
                os.path.join(self.data_dir, f"{symbol}_sample_data.csv"),
                os.path.join(self.data_dir, f"{base_name}_historical.csv"),
                os.path.join(self.data_dir, f"{symbol}_historical.csv")
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    logger.info(f"Loading development data from {file_path}")
                    df = pd.read_csv(file_path, parse_dates=['timestamp'])
                    
                    # Ensure required columns
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        df = df.sort_values('timestamp')
                        df.set_index('timestamp', inplace=True)
                        
                        # Add symbol attribute for risk management
                        df.attrs['symbol'] = symbol
                        
                        return df
                    else:
                        logger.warning(f"Missing required columns in {file_path}")
            
            logger.error(f"No development data found for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading development data for {symbol}: {e}")
            return None
    
    def _load_live_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from live environment."""
        try:
            # Live data format from download_real_data.py
            file_paths = [
                os.path.join(self.data_dir, f"{symbol}_live_data.csv"),
                os.path.join(self.data_dir, f"{symbol}_real_data.csv"),
                os.path.join(self.data_dir, f"{symbol}_historical.csv")
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    logger.info(f"Loading live data from {file_path}")
                    df = pd.read_csv(file_path, parse_dates=['timestamp'])
                    
                    # Ensure required columns
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        df = df.sort_values('timestamp')
                        df.set_index('timestamp', inplace=True)
                        
                        # Add symbol attribute
                        df.attrs['symbol'] = symbol
                        
                        # For live data, only use data up to a certain point to avoid look-ahead bias
                        cutoff_date = datetime.now() - timedelta(days=1)
                        df = df[df.index < cutoff_date]
                        
                        return df
                    else:
                        logger.warning(f"Missing required columns in {file_path}")
            
            logger.error(f"No live data found for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading live data for {symbol}: {e}")
            return None
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with all necessary features."""
        try:
            # Basic preprocessing
            df = self.preprocessor.prepare_ohlcv_data(df)
            
            # Calculate technical indicators
            df = self.preprocessor.calculate_technical_indicators(df)
            
            # Engineer advanced features
            df = self.feature_engineer.engineer_all_features(df)
            
            # Ensure we have volatility for risk management
            if 'volatility_20' not in df.columns:
                df['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
                df['volatility_20'].fillna(0.02, inplace=True)  # Default 2% volatility
            
            # Add microstructure features (simulated for historical data)
            if 'order_flow_imbalance' not in df.columns:
                df['order_flow_imbalance'] = np.random.normal(0, 0.1, len(df))
            if 'bid_ask_spread' not in df.columns:
                df['bid_ask_spread'] = df['close'] * 0.001  # 0.1% spread
            if 'pressure_ratio' not in df.columns:
                df['pressure_ratio'] = np.random.uniform(0.8, 1.2, len(df))
            if 'depth_imbalance' not in df.columns:
                df['depth_imbalance'] = np.random.normal(0, 0.2, len(df))
            if 'estimated_price_impact' not in df.columns:
                df['estimated_price_impact'] = np.random.uniform(0, 0.001, len(df))
            
            # Add alternative data features (simulated)
            if 'social_sentiment' not in df.columns:
                df['social_sentiment'] = np.random.normal(0, 0.5, len(df))
            if 'fear_greed_index' not in df.columns:
                df['fear_greed_index'] = np.random.uniform(20, 80, len(df))
            if 'whale_movements' not in df.columns:
                df['whale_movements'] = np.random.normal(0, 0.3, len(df))
            if 'exchange_flows' not in df.columns:
                df['exchange_flows'] = np.random.normal(0, 0.2, len(df))
            
            # Add multi-timeframe features
            if 'htf_trend' not in df.columns:
                df['htf_trend'] = df['close'].rolling(100).mean() > df['close'].rolling(200).mean()
            if 'mtf_trend' not in df.columns:
                df['mtf_trend'] = df['close'].rolling(20).mean() > df['close'].rolling(50).mean()
            if 'stf_trend' not in df.columns:
                df['stf_trend'] = df['close'].rolling(5).mean() > df['close'].rolling(10).mean()
            if 'trend_agreement' not in df.columns:
                df['trend_agreement'] = (df['htf_trend'].astype(int) + 
                                        df['mtf_trend'].astype(int) + 
                                        df['stf_trend'].astype(int)) / 3
            if 'momentum_divergence' not in df.columns:
                df['momentum_divergence'] = 0  # Simplified
            
            # Forward fill any NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)  # Fill any remaining NaNs with 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_agents(self, symbols: Optional[List[str]] = None):
        """Train RL agents for specified symbols."""
        if symbols is None:
            symbols = self.config.get('trading.symbols', ['BTC-USD'])
        
        logger.info(f"Starting RL training for symbols: {symbols}")
        
        for symbol in symbols:
            logger.info(f"\nTraining agents for {symbol}...")
            
            # Load data
            df = self.load_training_data(symbol)
            if df is None or len(df) < 1000:
                logger.error(f"Insufficient data for {symbol}, skipping...")
                continue
            
            # Prepare features
            df = self.prepare_data(df)
            
            # Create risk-aware environment
            env = RiskAwareTradingEnvironment(
                market_data=df,
                initial_balance=self.config.get('trading.initial_capital', 100000),
                risk_manager=self.risk_manager
            )
            
            # Store environment for reward functions
            self.rl_system.current_env = env
            
            # Train each specialized agent
            training_episodes = self.config.get('ml_models.rl_training_episodes', 1000)
            
            logger.info(f"Training {len(self.rl_system.agents)} agents for {training_episodes} episodes each...")
            
            # Custom training for each agent with risk-aware environment
            for agent_name in self.rl_system.agents.keys():
                logger.info(f"\nTraining {agent_name}...")
                
                # Create fresh environment for each agent
                agent_env = RiskAwareTradingEnvironment(
                    market_data=df.copy(),
                    initial_balance=self.config.get('trading.initial_capital', 100000),
                    risk_manager=RiskManager(self.config.get('trading.initial_capital', 100000))
                )
                
                # Store environment reference
                self.rl_system.current_env = agent_env
                
                # Bind appropriate reward function
                if agent_name == 'trend_follower':
                    agent_env._calculate_sophisticated_reward = lambda es, tc: self.rl_system._trend_following_reward(es, tc)
                elif agent_name == 'mean_reverter':
                    agent_env._calculate_sophisticated_reward = lambda es, tc: self.rl_system._mean_reversion_reward(es, tc)
                elif agent_name == 'volatility_trader':
                    agent_env._calculate_sophisticated_reward = lambda es, tc: self.rl_system._volatility_trading_reward(es, tc)
                elif agent_name == 'momentum_trader':
                    agent_env._calculate_sophisticated_reward = lambda es, tc: self.rl_system._momentum_trading_reward(es, tc)
                
                # Create SAC agent
                from models.reinforcement_learning.multi_agent_system import SACAgent
                agent = SACAgent(
                    state_size=agent_env.state_size,
                    action_size=agent_env.action_space,
                    continuous_action_dim=1,
                    hidden_dim=256,
                    device=self.rl_system.device
                )
                
                # Training loop with progress tracking
                episode_rewards = []
                best_reward = -float('inf')
                
                for episode in range(training_episodes):
                    state = agent_env.reset()
                    done = False
                    episode_reward = 0
                    steps = 0
                    
                    while not done and steps < 1000:  # Max 1000 steps per episode
                        action = agent.act(state)
                        next_state, reward, done, info = agent_env.step(action)
                        agent.remember(state, action, reward, next_state, done)
                        
                        state = next_state
                        episode_reward += reward
                        steps += 1
                        
                        # Train every step after initial exploration
                        if len(agent.memory) > 1000:
                            agent.train(batch_size=256)
                    
                    episode_rewards.append(episode_reward)
                    
                    # Track best model
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        # Save best model
                        best_path = os.path.join(self.models_dir, f"{agent_name}_best.pt")
                        agent.save(best_path)
                    
                    # Progress logging
                    if episode % 10 == 0:
                        avg_reward = np.mean(episode_rewards[-10:])
                        logger.info(
                            f"{agent_name} - Episode {episode}/{training_episodes}, "
                            f"Reward: {episode_reward:.2f}, Avg: {avg_reward:.2f}, "
                            f"Best: {best_reward:.2f}, "
                            f"Portfolio: ${info['portfolio_value']:.2f}, "
                            f"Epsilon: {agent.epsilon:.3f}"
                        )
                    
                    # Periodic saving
                    if episode % 100 == 0 and episode > 0:
                        checkpoint_path = os.path.join(
                            self.models_dir, 
                            f"{agent_name}_checkpoint_{episode}.pt"
                        )
                        agent.save(checkpoint_path)
                
                # Store trained agent and performance
                self.rl_system.agents[agent_name] = agent
                self.rl_system.agent_performance[agent_name] = episode_rewards
                
                # Final save
                final_path = os.path.join(self.models_dir, f"{agent_name}_final.pt")
                agent.save(final_path)
                
                logger.info(f"Completed training {agent_name}. Final avg reward: {np.mean(episode_rewards[-100:]):.2f}")
            
            # Save all agents
            self.rl_system.save_all_agents(self.models_dir)
            
            # Generate training report
            self._generate_training_report(symbol)
    
    def _generate_training_report(self, symbol: str):
        """Generate detailed training report."""
        try:
            report = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'environment': self.config.get('environment', 'development'),
                'training_episodes': self.config.get('ml_models.rl_training_episodes', 1000),
                'agents': {}
            }
            
            for agent_name, performance in self.rl_system.agent_performance.items():
                if len(performance) > 0:
                    report['agents'][agent_name] = {
                        'final_avg_reward': float(np.mean(performance[-100:])),
                        'best_reward': float(np.max(performance)),
                        'total_improvement': float(np.mean(performance[-100:]) - np.mean(performance[:100])),
                        'convergence_episode': self._find_convergence_episode(performance)
                    }
            
            # Save report
            report_path = os.path.join(self.models_dir, f"training_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating training report: {e}")
    
    def _find_convergence_episode(self, rewards: List[float], window: int = 50, threshold: float = 0.05) -> int:
        """Find episode where training converged."""
        if len(rewards) < window * 2:
            return len(rewards)
        
        for i in range(window, len(rewards) - window):
            prev_avg = np.mean(rewards[i-window:i])
            curr_avg = np.mean(rewards[i:i+window])
            
            if abs(curr_avg - prev_avg) / (abs(prev_avg) + 1e-8) < threshold:
                return i
        
        return len(rewards)
    
    def evaluate_agents(self, symbol: str, test_days: int = 30):
        """Evaluate trained agents on recent data."""
        logger.info(f"\nEvaluating agents on {symbol} with {test_days} days of test data...")
        
        # Load test data
        df = self.load_training_data(symbol)
        if df is None:
            logger.error(f"No data available for evaluation")
            return
        
        # Use last N days for testing
        df = self.prepare_data(df)
        test_data = df.iloc[-test_days*24:]  # Assuming hourly data
        
        if len(test_data) < 100:
            logger.error(f"Insufficient test data")
            return
        
        results = {}
        
        for agent_name, agent in self.rl_system.agents.items():
            if agent is None:
                continue
            
            # Create test environment
            test_env = RiskAwareTradingEnvironment(
                market_data=test_data.copy(),
                initial_balance=self.config.get('trading.initial_capital', 100000),
                risk_manager=RiskManager(self.config.get('trading.initial_capital', 100000))
            )
            
            # Run evaluation
            state = test_env.reset()
            done = False
            total_reward = 0
            trades = 0
            
            while not done:
                action = agent.predict(state)  # Use predict (no exploration)
                next_state, reward, done, info = test_env.step(action)
                
                if action != 0:  # Not hold
                    trades += 1
                
                total_reward += reward
                state = next_state
            
            # Calculate metrics
            final_value = test_env.get_portfolio_value()
            total_return = (final_value - test_env.initial_balance) / test_env.initial_balance
            
            results[agent_name] = {
                'total_return': total_return,
                'total_reward': total_reward,
                'num_trades': trades,
                'final_portfolio_value': final_value,
                'sharpe_ratio': self._calculate_sharpe(test_env),
                'max_drawdown': test_env.max_drawdown
            }
            
            logger.info(f"{agent_name}: Return={total_return:.2%}, Trades={trades}, Sharpe={results[agent_name]['sharpe_ratio']:.2f}")
        
        # Save evaluation results
        eval_path = os.path.join(self.models_dir, f"evaluation_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_path}")
    
    def _calculate_sharpe(self, env) -> float:
        """Calculate Sharpe ratio from environment trades."""
        if len(env.trades) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(env.trades)):
            if env.trades[i]['type'] != env.trades[i-1]['type']:
                ret = (env.trades[i]['price'] - env.trades[i-1]['price']) / env.trades[i-1]['price']
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train RL agents for cryptocurrency trading')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on (e.g., BTC-USD ETH-USD)')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate agents after training')
    parser.add_argument('--test-days', type=int, default=30, help='Days of data for evaluation')
    
    args = parser.parse_args()
    
    # Create training manager
    manager = RLTrainingManager(args.config)
    
    # Override config with command line args
    if args.episodes:
        manager.config['ml_models']['rl_training_episodes'] = args.episodes
    
    # Create models directory
    os.makedirs(manager.models_dir, exist_ok=True)
    
    # Train agents
    symbols = args.symbols or manager.config.get('trading.symbols', ['BTC-USD'])
    manager.train_agents(symbols)
    
    # Evaluate if requested
    if args.evaluate:
        for symbol in symbols:
            manager.evaluate_agents(symbol, args.test_days)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
