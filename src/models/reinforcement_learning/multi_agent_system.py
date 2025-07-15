"""
Reinforcement Learning Trading System for Cryptocurrency

This module implements an advanced RL-based trading system with:
- Sophisticated trading environment with enhanced state space
- Multi-agent ensemble system with specialized trading strategies
- Advanced reward functions for different market conditions
- Risk management and position sizing controls
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RLAction:
    """Data class representing a trading action with metadata."""
    action_type: str  # 'hold', 'buy', 'sell'
    intensity: float  # 0.0 to 1.0
    confidence: float
    expected_reward: float


class CryptoTradingEnvironment:
    """
    Advanced RL environment for cryptocurrency trading.
    
    Features:
    - Enhanced state space with microstructure and alternative data
    - Sophisticated reward function with multiple components
    - Risk management controls
    - Transaction cost modeling
    
    Attributes:
        market_data: Historical market data DataFrame
        initial_balance: Starting capital
        action_space: Number of possible actions (7)
        state_size: Size of the state vector (75)
    """
    
    def __init__(self, market_data: pd.DataFrame, initial_balance: float = 100000):
        """
        Initialize the trading environment.
        
        Args:
            market_data: DataFrame with OHLCV and indicator data
            initial_balance: Starting capital amount
        """
        self.market_data = market_data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        self.max_drawdown = 0
        self.peak_value = initial_balance
        
        # Enhanced action space: [hold, buy_25%, buy_50%, buy_100%, sell_25%, sell_50%, sell_100%]
        self.action_space = 7
        
        # Enhanced state space with microstructure and alternative data
        self.state_size = 75
        
        # Risk management parameters
        self.max_position_size = 0.95  # Max 95% of capital
        self.transaction_cost = 0.001  # 0.1% transaction cost
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state vector
        """
        self.current_step = 60  # Start after warmup period
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.max_drawdown = 0
        self.peak_value = self.initial_balance
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state.
        
        Args:
            action: Integer action (0-6)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Execute action
        reward = self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        current_value = self.get_portfolio_value()
        drawdown = (self.peak_value - current_value) / self.peak_value
        
        done = (
            self.current_step >= len(self.market_data) - 1 or 
            drawdown > 0.5 or  # 50% drawdown limit
            current_value <= self.initial_balance * 0.1  # 90% loss limit
        )
        
        next_state = self._get_state() if not done else np.zeros(self.state_size)
        
        return next_state, reward, done, self._get_info()
    
    def _get_state(self) -> np.ndarray:
        """
        Get comprehensive market state with enhanced features.
        
        Returns:
            State vector of size self.state_size
        """
        if self.current_step >= len(self.market_data):
            return np.zeros(self.state_size)
        
        # Price features (last 20 periods)
        price_window = self.market_data['close'][
            max(0, self.current_step-20):self.current_step
        ]
        if len(price_window) < 20:
            price_window = np.pad(
                price_window, 
                (20-len(price_window), 0), 
                'edge'
            )
        
        price_returns = np.diff(np.log(price_window + 1e-8))
        
        # Technical indicators
        current_data = self.market_data.iloc[self.current_step]
        technical_features = [
            current_data.get('rsi_14', 50) / 100,  # Normalized RSI
            current_data.get('macd', 0) / 100,      # Normalized MACD
            current_data.get('bb_position', 0.5),   # Bollinger Band position
            current_data.get('atr', 0) / current_data.get('close', 1),  # Normalized ATR
            current_data.get('volume_ratio', 1),    # Volume ratio
        ]
        
        # Market microstructure (enhanced)
        microstructure_features = [
            current_data.get('order_flow_imbalance', 0),
            current_data.get('bid_ask_spread', 0) / current_data.get('close', 1),
            current_data.get('pressure_ratio', 1),
            current_data.get('depth_imbalance', 0),
            current_data.get('estimated_price_impact', 0)
        ]
        
        # Alternative data features
        alt_data_features = [
            current_data.get('social_sentiment', 0),
            current_data.get('fear_greed_index', 50) / 100,
            current_data.get('whale_movements', 0),
            current_data.get('exchange_flows', 0)
        ]
        
        # Portfolio state
        current_price = current_data.get('close', price_window[-1])
        portfolio_value = self.get_portfolio_value()
        position_ratio = (
            (self.position * current_price) / portfolio_value 
            if portfolio_value > 0 else 0
        )
        cash_ratio = self.balance / portfolio_value if portfolio_value > 0 else 1
        
        # Performance metrics
        total_return = (
            (portfolio_value - self.initial_balance) / self.initial_balance
        )
        current_drawdown = (
            (self.peak_value - portfolio_value) / self.peak_value 
            if self.peak_value > 0 else 0
        )
        
        # Market regime features
        volatility = (
            np.std(price_returns[-10:]) 
            if len(price_returns) >= 10 else 0
        )
        trend = (
            (current_price - price_window[0]) / price_window[0] 
            if price_window[0] > 0 else 0
        )
        momentum = (
            np.mean(price_returns[-5:]) 
            if len(price_returns) >= 5 else 0
        )
        
        # Multi-timeframe features
        mtf_features = [
            current_data.get('htf_trend', False),
            current_data.get('mtf_trend', False),
            current_data.get('stf_trend', False),
            current_data.get('trend_agreement', 0),
            current_data.get('momentum_divergence', 0)
        ]
        
        # Recent trading performance
        recent_trades = self.trades[-10:] if len(self.trades) >= 10 else self.trades
        if recent_trades:
            avg_trade_return = np.mean([t.get('return', 0) for t in recent_trades])
            trade_win_rate = np.mean([
                1 if t.get('return', 0) > 0 else 0 for t in recent_trades
            ])
        else:
            avg_trade_return = 0
            trade_win_rate = 0.5
        
        # Combine all features
        state = np.concatenate([
            price_returns[-15:],  # 15 recent returns
            technical_features,   # 5 technical indicators
            microstructure_features,  # 5 microstructure features
            alt_data_features,    # 4 alternative data features
            mtf_features,         # 5 multi-timeframe features
            [
                position_ratio, 
                cash_ratio, 
                total_return, 
                current_drawdown,
                volatility, 
                trend, 
                momentum, 
                avg_trade_return, 
                trade_win_rate,
                len(self.trades) / 1000,  # Normalized trade count
                self.current_step / len(self.market_data)  # Progress through data
            ]  # 11 portfolio/performance features
        ])
        
        # Pad or truncate to exact state size
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)), 'constant')
        else:
            state = state[:self.state_size]
        
        return state.astype(np.float32)
    
    def _execute_action(self, action: int) -> float:
        """
        Execute trading action with enhanced logic.
        
        Args:
            action: Integer action to execute
            
        Returns:
            Reward value
        """
        current_price = self.market_data['close'].iloc[self.current_step]
        
        # Define action mappings with risk controls
        if action == 0:  # Hold
            trade_size = 0
        elif action == 1:  # Buy 25%
            available_capital = self.balance * 0.25
            max_position_value = self.initial_balance * self.max_position_size
            current_position_value = self.position * current_price
            max_buy_value = max_position_value - current_position_value
            trade_size = min(
                available_capital / current_price,
                max_buy_value / current_price
            )
        elif action == 2:  # Buy 50%
            available_capital = self.balance * 0.50
            max_position_value = self.initial_balance * self.max_position_size
            current_position_value = self.position * current_price
            max_buy_value = max_position_value - current_position_value
            trade_size = min(
                available_capital / current_price,
                max_buy_value / current_price
            )
        elif action == 3:  # Buy 100%
            available_capital = self.balance
            max_position_value = self.initial_balance * self.max_position_size
            current_position_value = self.position * current_price
            max_buy_value = max_position_value - current_position_value
            trade_size = min(
                available_capital / current_price,
                max_buy_value / current_price
            )
        elif action == 4:  # Sell 25%
            trade_size = -self.position * 0.25
        elif action == 5:  # Sell 50%
            trade_size = -self.position * 0.50
        else:  # Sell 100%
            trade_size = -self.position
        
        # Execute trade with transaction costs
        executed_size = 0
        transaction_cost = 0
        
        if trade_size > 0:  # Buy
            cost = trade_size * current_price
            transaction_cost = cost * self.transaction_cost
            total_cost = cost + transaction_cost
            
            if total_cost <= self.balance and trade_size > 0.001:  # Minimum trade size
                self.balance -= total_cost
                self.position += trade_size
                executed_size = trade_size
                
                self.trades.append({
                    'type': 'buy',
                    'size': trade_size,
                    'price': current_price,
                    'cost': transaction_cost,
                    'timestamp': self.current_step
                })
                
        elif trade_size < 0:  # Sell
            sell_amount = abs(trade_size)
            if sell_amount <= self.position and sell_amount > 0.001:  # Minimum trade size
                revenue = sell_amount * current_price
                transaction_cost = revenue * self.transaction_cost
                net_revenue = revenue - transaction_cost
                
                self.balance += net_revenue
                self.position -= sell_amount
                executed_size = trade_size
                
                self.trades.append({
                    'type': 'sell',
                    'size': sell_amount,
                    'price': current_price,
                    'cost': transaction_cost,
                    'timestamp': self.current_step
                })
        
        # Calculate reward
        reward = self._calculate_sophisticated_reward(executed_size, transaction_cost)
        
        # Update peak value and drawdown
        current_value = self.get_portfolio_value()
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        return reward
    
    def _calculate_sophisticated_reward(
        self, 
        executed_size: float, 
        transaction_cost: float
    ) -> float:
        """
        Calculate sophisticated reward function for RL training.
        
        Args:
            executed_size: Size of executed trade
            transaction_cost: Cost of transaction
            
        Returns:
            Calculated reward value
        """
        current_value = self.get_portfolio_value()
        
        # Base portfolio return component
        portfolio_return = (
            (current_value - self.initial_balance) / self.initial_balance
        )
        return_component = portfolio_return * 100  # Scale up
        
        # Risk-adjusted return (Sharpe ratio component)
        if len(self.trades) >= 20:
            trade_returns = self._calculate_recent_trade_returns()
            if len(trade_returns) > 1:
                mean_return = np.mean(trade_returns)
                volatility = np.std(trade_returns)
                sharpe_component = (mean_return / (volatility + 1e-8)) * 10  # Scale up
            else:
                sharpe_component = 0
        else:
            sharpe_component = 0
        
        # Drawdown penalty (progressive)
        current_drawdown = (
            (self.peak_value - current_value) / self.peak_value 
            if self.peak_value > 0 else 0
        )
        if current_drawdown > 0.05:  # 5% threshold
            drawdown_penalty = -50 * (current_drawdown - 0.05) ** 2  # Quadratic penalty
        else:
            drawdown_penalty = 0
        
        # Transaction cost penalty
        cost_penalty = -transaction_cost / self.initial_balance * 1000  # Scale penalty
        
        # Position management reward
        position_value = self.position * self.market_data['close'].iloc[self.current_step]
        portfolio_balance = position_value + self.balance
        if portfolio_balance > 0:
            position_ratio = position_value / portfolio_balance
            # Reward balanced positions, penalize extreme allocations
            if 0.3 <= position_ratio <= 0.8:
                position_reward = 2
            elif position_ratio > 0.95 or position_ratio < 0.05:
                position_reward = -5
            else:
                position_reward = 0
        else:
            position_reward = -10
        
        # Trading frequency management
        recent_trades = len([
            t for t in self.trades 
            if self.current_step - t['timestamp'] <= 20
        ])
        if recent_trades > 10:  # More than 10 trades in 20 periods
            frequency_penalty = -2 * (recent_trades - 10)
        else:
            frequency_penalty = 0
        
        # Momentum alignment reward
        if len(self.trades) > 0:
            last_trade = self.trades[-1]
            current_price = self.market_data['close'].iloc[self.current_step]
            
            if last_trade['type'] == 'buy':
                if current_price > last_trade['price']:
                    momentum_reward = 1  # Bought before price increase
                else:
                    momentum_reward = -0.5
            elif last_trade['type'] == 'sell':
                if current_price < last_trade['price']:
                    momentum_reward = 1  # Sold before price decrease
                else:
                    momentum_reward = -0.5
            else:
                momentum_reward = 0
        else:
            momentum_reward = 0
        
        # Combine all reward components
        total_reward = (
            return_component + 
            sharpe_component + 
            drawdown_penalty + 
            cost_penalty + 
            position_reward + 
            frequency_penalty + 
            momentum_reward
        )
        
        return total_reward
    
    def _calculate_recent_trade_returns(self, window: int = 20) -> List[float]:
        """
        Calculate returns for recent trades.
        
        Args:
            window: Number of recent trades to consider
            
        Returns:
            List of trade returns
        """
        if len(self.trades) < 2:
            return []
        
        returns = []
        recent_trades = self.trades[-window:]
        
        # Simple return calculation based on price changes
        for i in range(1, len(recent_trades)):
            prev_trade = recent_trades[i-1]
            curr_trade = recent_trades[i]
            
            if prev_trade['type'] == 'buy' and curr_trade['type'] == 'sell':
                trade_return = (
                    (curr_trade['price'] - prev_trade['price']) / prev_trade['price']
                )
                returns.append(trade_return)
        
        return returns
    
    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value.
        
        Returns:
            Total portfolio value (cash + position value)
        """
        if self.current_step < len(self.market_data):
            current_price = self.market_data['close'].iloc[self.current_step]
        else:
            current_price = self.market_data['close'].iloc[-1]
        
        return self.balance + self.position * current_price
    
    def _get_info(self) -> Dict:
        """
        Get additional information for logging/analysis.
        
        Returns:
            Dict with portfolio metrics
        """
        return {
            'portfolio_value': self.get_portfolio_value(),
            'position': self.position,
            'balance': self.balance,
            'num_trades': len(self.trades),
            'max_drawdown': self.max_drawdown
        }


class MultiAgentTradingSystem:
    """
    Multi-agent system with specialized agents for different strategies.
    
    This system combines multiple specialized agents:
    - Trend following agent
    - Mean reversion agent
    - Volatility trading agent
    - Momentum trading agent
    
    A meta-agent dynamically allocates capital based on market conditions
    and agent performance.
    """
    
    def __init__(self):
        """Initialize the multi-agent system."""
        # Specialized agents for different strategies
        self.agents = {
            'trend_follower': None,
            'mean_reverter': None,
            'volatility_trader': None,
            'momentum_trader': None
        }
        
        # Meta-agent for allocation decisions
        self.meta_agent = None
        self.agent_performance = {name: [] for name in self.agents.keys()}
    
    def train_agents(
        self, 
        market_data: pd.DataFrame, 
        training_episodes: int = 1000
    ) -> None:
        """
        Train specialized agents using different reward functions.
        
        Args:
            market_data: Historical market data for training
            training_episodes: Number of episodes per agent
        """
        logger.info("Training multi-agent system...")
        
        for agent_name in self.agents.keys():
            logger.info(f"Training {agent_name}...")
            
            # Create environment with agent-specific modifications
            env = CryptoTradingEnvironment(market_data)
            
            # Customize reward function for each agent type
            if agent_name == 'trend_follower':
                env._calculate_sophisticated_reward = self._trend_following_reward
            elif agent_name == 'mean_reverter':
                env._calculate_sophisticated_reward = self._mean_reversion_reward
            elif agent_name == 'volatility_trader':
                env._calculate_sophisticated_reward = self._volatility_trading_reward
            elif agent_name == 'momentum_trader':
                env._calculate_sophisticated_reward = self._momentum_trading_reward
            
            # Simple neural network agent
            agent = SimpleNeuralAgent(env.state_size, env.action_space)
            
            # Train agent
            for episode in range(training_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward
                
                self.agent_performance[agent_name].append(episode_reward)
                
                # Train agent periodically
                if episode % 50 == 0:
                    agent.train()
            
            self.agents[agent_name] = agent
            logger.info(f"Completed training {agent_name}")
    
    def get_ensemble_action(
        self, 
        state: np.ndarray, 
        market_conditions: Dict
    ) -> int:
        """
        Get weighted ensemble action from all agents.
        
        Args:
            state: Current market state
            market_conditions: Dict with market regime information
            
        Returns:
            Ensemble action (0-6)
        """
        if not all(self.agents.values()):
            return 0  # Default to hold if agents not trained
        
        agent_actions = {}
        agent_confidences = {}
        
        for agent_name, agent in self.agents.items():
            if agent is not None:
                action = agent.predict(state)
                confidence = agent.get_confidence(state)
                agent_actions[agent_name] = action
                agent_confidences[agent_name] = confidence
        
        # Calculate dynamic weights based on recent performance and market conditions
        weights = self._calculate_agent_weights(market_conditions)
        
        # Weighted voting with confidence adjustment
        action_scores = {}
        for agent_name, action in agent_actions.items():
            weight = weights.get(agent_name, 0)
            confidence = agent_confidences.get(agent_name, 0.5)
            
            adjusted_weight = weight * confidence
            
            if action not in action_scores:
                action_scores[action] = 0
            action_scores[action] += adjusted_weight
        
        # Return action with highest weighted score
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            return best_action
        
        return 0  # Default to hold
    
    def _calculate_agent_weights(self, market_conditions: Dict) -> Dict[str, float]:
        """
        Calculate dynamic weights for agents based on market conditions.
        
        Args:
            market_conditions: Dict with market regime information
            
        Returns:
            Dict of agent weights
        """
        # Get recent performance
        recent_window = 50
        performance_scores = {}
        
        for agent_name, performance in self.agent_performance.items():
            if len(performance) >= recent_window:
                recent_perf = performance[-recent_window:]
                avg_performance = np.mean(recent_perf)
                performance_scores[agent_name] = max(0, avg_performance)
            else:
                performance_scores[agent_name] = 0
        
        # Market condition adjustments
        volatility = market_conditions.get('volatility', 0.02)
        trend_strength = market_conditions.get('trend_strength', 0)
        
        # Base weights
        weights = {
            'trend_follower': 0.25,
            'mean_reverter': 0.25,
            'volatility_trader': 0.25,
            'momentum_trader': 0.25
        }
        
        # Adjust based on market conditions
        if volatility > 0.04:  # High volatility
            weights['volatility_trader'] *= 1.5
            weights['mean_reverter'] *= 0.7
        elif volatility < 0.015:  # Low volatility
            weights['mean_reverter'] *= 1.4
            weights['volatility_trader'] *= 0.6
        
        if abs(trend_strength) > 0.02:  # Strong trend
            weights['trend_follower'] *= 1.3
            weights['momentum_trader'] *= 1.2
            weights['mean_reverter'] *= 0.6
        
        # Adjust based on recent performance
        total_performance = sum(performance_scores.values())
        if total_performance > 0:
            for agent_name in weights.keys():
                performance_weight = performance_scores[agent_name] / total_performance
                weights[agent_name] = (
                    weights[agent_name] * 0.7 + performance_weight * 0.3
                )
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _trend_following_reward(
        self, 
        executed_size: float, 
        transaction_cost: float
    ) -> float:
        """
        Reward function optimized for trend following.
        
        Note: This is a placeholder - implement actual trend-following
        reward logic based on position alignment with trend.
        """
        # TODO: Implement trend-following specific reward
        return 0
    
    def _mean_reversion_reward(
        self, 
        executed_size: float, 
        transaction_cost: float
    ) -> float:
        """
        Reward function optimized for mean reversion.
        
        Note: This is a placeholder - implement actual mean reversion
        reward logic based on buying oversold and selling overbought.
        """
        # TODO: Implement mean reversion specific reward
        return 0
    
    def _volatility_trading_reward(
        self, 
        executed_size: float, 
        transaction_cost: float
    ) -> float:
        """
        Reward function optimized for volatility trading.
        
        Note: This is a placeholder - implement actual volatility trading
        reward logic based on volatility regime.
        """
        # TODO: Implement volatility trading specific reward
        return 0
    
    def _momentum_trading_reward(
        self, 
        executed_size: float, 
        transaction_cost: float
    ) -> float:
        """
        Reward function optimized for momentum trading.
        
        Note: This is a placeholder - implement actual momentum trading
        reward logic based on momentum indicators.
        """
        # TODO: Implement momentum trading specific reward
        return 0


class SimpleNeuralAgent:
    """
    Simple neural network agent for RL.
    
    This is a basic DQN-style agent implementation for trading.
    In production, consider using more advanced algorithms like:
    - PPO (Proximal Policy Optimization)
    - SAC (Soft Actor-Critic)
    - TD3 (Twin Delayed DDPG)
    
    Attributes:
        state_size: Size of input state vector
        action_size: Number of possible actions
        epsilon: Exploration rate
        memory: Experience replay buffer
    """
    
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize the neural agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Simple neural network
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def remember(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode finished flag
        """
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return q_values.argmax().item()
    
    def predict(self, state: np.ndarray) -> int:
        """
        Predict action without exploration.
        
        Args:
            state: Current state
            
        Returns:
            Best action according to Q-values
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()
    
    def get_confidence(self, state: np.ndarray) -> float:
        """
        Get confidence in prediction.
        
        Args:
            state: Current state
            
        Returns:
            Confidence score (0-1)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            confidence = torch.softmax(q_values, dim=1).max().item()
        return confidence
    
    def train(self, batch_size: int = 32) -> None:
        """
        Train the neural network using experience replay.
        
        Args:
            batch_size: Number of experiences to sample
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = torch.FloatTensor([self.memory[i][0] for i in batch_indices])
        actions = torch.LongTensor([self.memory[i][1] for i in batch_indices])
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch_indices])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch_indices])
        dones = torch.BoolTensor([self.memory[i][4] for i in batch_indices])
        
        # Calculate current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Calculate target Q values
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Calculate loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay