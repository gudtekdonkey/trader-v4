"""
Cryptocurrency trading environment for reinforcement learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import traceback
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class CryptoTradingEnvironment:
    """
    Advanced RL environment for cryptocurrency trading with comprehensive error handling.
    
    Features:
    - Enhanced state space with microstructure and alternative data
    - Sophisticated reward function with multiple components
    - Risk management controls
    - Transaction cost modeling
    - Comprehensive error handling and state validation
    
    Attributes:
        market_data: Historical market data DataFrame
        initial_balance: Starting capital
        action_space: Number of possible actions (7)
        state_size: Size of the state vector (75)
    """
    
    def __init__(self, market_data: pd.DataFrame, initial_balance: float = 100000):
        """
        Initialize the trading environment with validation.
        
        Args:
            market_data: DataFrame with OHLCV and indicator data
            initial_balance: Starting capital amount
        """
        # [ERROR-HANDLING] Validate inputs
        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            raise ValueError("Invalid market data: must be non-empty DataFrame")
        
        if not isinstance(initial_balance, (int, float)) or initial_balance <= 0:
            raise ValueError(f"Invalid initial balance: {initial_balance}")
        
        # [ERROR-HANDLING] Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.market_data = market_data.copy()  # Defensive copy
        self.initial_balance = float(initial_balance)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.max_drawdown = 0
        self.peak_value = self.initial_balance
        
        # Enhanced action space
        self.action_space = 7
        self.state_size = 75
        
        # Risk management parameters
        self.max_position_size = 0.95
        self.transaction_cost = 0.001
        
        # [ERROR-HANDLING] State validation tracking
        self.state_errors = 0
        self.max_state_errors = 100
        
        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
        
        logger.info(f"Environment initialized with {len(market_data)} data points")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state with error handling.
        
        Returns:
            Initial state vector
        """
        try:
            self.current_step = 60  # Start after warmup period
            self.balance = self.initial_balance
            self.position = 0
            self.trades = []
            self.max_drawdown = 0
            self.peak_value = self.initial_balance
            self.episode_count += 1
            
            # [ERROR-HANDLING] Validate we have enough data
            if self.current_step >= len(self.market_data):
                logger.warning("Not enough data for warmup period, adjusting start")
                self.current_step = min(60, len(self.market_data) - 1)
            
            state = self._get_state()
            
            # [ERROR-HANDLING] Validate state
            if not self._validate_state(state):
                logger.error("Invalid initial state generated")
                return np.zeros(self.state_size, dtype=np.float32)
            
            return state
            
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            return np.zeros(self.state_size, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state with comprehensive error handling.
        
        Args:
            action: Integer action (0-6)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        try:
            # [ERROR-HANDLING] Validate action
            if not isinstance(action, (int, np.integer)) or action < 0 or action >= self.action_space:
                logger.warning(f"Invalid action: {action}, defaulting to hold")
                action = 0
            
            # [ERROR-HANDLING] Check if we can continue
            if self.current_step >= len(self.market_data) - 1:
                logger.warning("Reached end of data")
                return self._get_terminal_state()
            
            # Execute action with error handling
            reward = self._execute_action_safe(action)
            
            # Move to next step
            self.current_step += 1
            self.total_steps += 1
            
            # Calculate portfolio metrics
            current_value = self.get_portfolio_value()
            
            # [ERROR-HANDLING] Validate portfolio value
            if not np.isfinite(current_value) or current_value < 0:
                logger.error(f"Invalid portfolio value: {current_value}")
                return self._get_terminal_state()
            
            # Update drawdown
            if current_value > self.peak_value:
                self.peak_value = current_value
            
            drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Check termination conditions
            done = self._check_done_conditions(current_value, drawdown)
            
            # Get next state
            if done:
                next_state = np.zeros(self.state_size, dtype=np.float32)
            else:
                next_state = self._get_state()
                
                # [ERROR-HANDLING] Validate state
                if not self._validate_state(next_state):
                    self.state_errors += 1
                    if self.state_errors > self.max_state_errors:
                        logger.error("Too many state errors, terminating episode")
                        done = True
                        next_state = np.zeros(self.state_size, dtype=np.float32)
            
            # Compile info dict
            info = self._get_info()
            
            return next_state, reward, done, info
            
        except Exception as e:
            logger.error(f"Critical error in step: {e}")
            logger.error(traceback.format_exc())
            return self._get_terminal_state()
    
    def _get_terminal_state(self) -> Tuple[np.ndarray, float, bool, Dict]:
        """Return terminal state for error conditions"""
        return (
            np.zeros(self.state_size, dtype=np.float32),
            -10.0,  # Penalty reward
            True,   # Done
            {'error': 'terminal_state', 'portfolio_value': self.get_portfolio_value()}
        )
    
    def _check_done_conditions(self, current_value: float, drawdown: float) -> bool:
        """Check if episode should terminate"""
        # Normal termination
        if self.current_step >= len(self.market_data) - 1:
            return True
        
        # Risk-based termination
        if drawdown > 0.5:  # 50% drawdown
            logger.warning("Episode terminated: Maximum drawdown exceeded")
            return True
        
        if current_value <= self.initial_balance * 0.1:  # 90% loss
            logger.warning("Episode terminated: Excessive losses")
            return True
        
        # [ERROR-HANDLING] Check for degenerate states
        if self.balance < 0:
            logger.error("Negative balance detected")
            return True
        
        if abs(self.position) * self.market_data['close'].iloc[self.current_step] > current_value * 2:
            logger.error("Position size exceeds reasonable limits")
            return True
        
        return False
    
    def _validate_state(self, state: np.ndarray) -> bool:
        """Validate state vector"""
        try:
            if not isinstance(state, np.ndarray):
                return False
            
            if state.shape != (self.state_size,):
                logger.error(f"Invalid state shape: {state.shape}")
                return False
            
            if not np.all(np.isfinite(state)):
                logger.warning("Non-finite values in state")
                return False
            
            # Check for reasonable ranges
            if np.any(np.abs(state) > 100):
                logger.warning("Extreme values in state vector")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating state: {e}")
            return False
    
    def _get_state(self) -> np.ndarray:
        """
        Get comprehensive market state with enhanced features and error handling.
        
        Returns:
            State vector of size self.state_size
        """
        try:
            if self.current_step >= len(self.market_data):
                return np.zeros(self.state_size, dtype=np.float32)
            
            # Price features with validation
            price_window_size = 20
            start_idx = max(0, self.current_step - price_window_size)
            price_window = self.market_data['close'][start_idx:self.current_step].values
            
            # [ERROR-HANDLING] Handle insufficient data
            if len(price_window) < price_window_size:
                price_window = np.pad(price_window, (price_window_size - len(price_window), 0), 'edge')
            
            # [ERROR-HANDLING] Validate prices
            if np.any(price_window <= 0):
                logger.warning("Non-positive prices detected")
                price_window = np.maximum(price_window, 1e-8)
            
            # Safe log returns calculation
            price_returns = np.diff(np.log(price_window + 1e-8))
            
            # [ERROR-HANDLING] Handle extreme returns
            price_returns = np.clip(price_returns, -0.5, 0.5)  # Cap at ±50%
            
            # Get current data with validation
            current_data = self.market_data.iloc[self.current_step]
            
            # Technical indicators with defaults
            technical_features = []
            indicator_defaults = {
                'rsi_14': 50, 'macd': 0, 'bb_position': 0.5,
                'atr': 0.02, 'volume_ratio': 1
            }
            
            for indicator, default in indicator_defaults.items():
                value = current_data.get(indicator, default)
                # Normalize and validate
                if indicator == 'rsi_14':
                    value = np.clip(value, 0, 100) / 100
                elif indicator == 'bb_position':
                    value = np.clip(value, 0, 1)
                elif indicator == 'atr':
                    value = value / current_data.get('close', 1)
                    value = np.clip(value, 0, 0.1)  # Cap at 10%
                elif indicator == 'volume_ratio':
                    value = np.clip(value, 0, 10)  # Cap at 10x
                else:
                    value = np.clip(value / 100, -1, 1)  # General normalization
                
                technical_features.append(float(value))
            
            # Market microstructure with error handling
            microstructure_features = []
            micro_defaults = {
                'order_flow_imbalance': 0, 'bid_ask_spread': 0.001,
                'pressure_ratio': 1, 'depth_imbalance': 0,
                'estimated_price_impact': 0
            }
            
            for feature, default in micro_defaults.items():
                value = current_data.get(feature, default)
                if feature == 'bid_ask_spread':
                    value = value / current_data.get('close', 1)
                    value = np.clip(value, 0, 0.01)  # Cap at 1%
                else:
                    value = np.clip(value, -1, 1)
                
                microstructure_features.append(float(value))
            
            # Alternative data features with validation
            alt_data_features = []
            alt_defaults = {
                'social_sentiment': 0, 'fear_greed_index': 50,
                'whale_movements': 0, 'exchange_flows': 0
            }
            
            for feature, default in alt_defaults.items():
                value = current_data.get(feature, default)
                if feature == 'fear_greed_index':
                    value = np.clip(value, 0, 100) / 100
                else:
                    value = np.clip(value, -1, 1)
                
                alt_data_features.append(float(value))
            
            # Portfolio state with safety checks
            current_price = current_data.get('close', price_window[-1] if len(price_window) > 0 else 1)
            if current_price <= 0:
                current_price = 1
                
            portfolio_value = self.get_portfolio_value()
            if portfolio_value <= 0:
                portfolio_value = 1
            
            position_value = self.position * current_price
            position_ratio = np.clip(position_value / portfolio_value, -1, 1)
            cash_ratio = np.clip(self.balance / portfolio_value, 0, 1)
            
            # Performance metrics with bounds
            total_return = (portfolio_value - self.initial_balance) / self.initial_balance
            total_return = np.clip(total_return, -1, 10)  # Cap at 1000% gain
            
            current_drawdown = np.clip(self.max_drawdown, 0, 1)
            
            # Market regime features with safety
            if len(price_returns) >= 10:
                volatility = np.std(price_returns[-10:])
                volatility = np.clip(volatility, 0, 0.1)  # Cap at 10%
            else:
                volatility = 0.02
            
            if price_window[0] > 0:
                trend = (current_price - price_window[0]) / price_window[0]
                trend = np.clip(trend, -1, 1)
            else:
                trend = 0
            
            if len(price_returns) >= 5:
                momentum = np.mean(price_returns[-5:])
                momentum = np.clip(momentum, -0.1, 0.1)
            else:
                momentum = 0
            
            # Multi-timeframe features with defaults
            mtf_features = []
            mtf_defaults = {
                'htf_trend': 0, 'mtf_trend': 0, 'stf_trend': 0,
                'trend_agreement': 0, 'momentum_divergence': 0
            }
            
            for feature, default in mtf_defaults.items():
                value = current_data.get(feature, default)
                if isinstance(value, bool):
                    value = float(value)
                else:
                    value = np.clip(float(value), -1, 1)
                mtf_features.append(value)
            
            # Trading performance with safety
            recent_trades = self.trades[-10:] if len(self.trades) >= 10 else self.trades
            if recent_trades:
                trade_returns = [t.get('return', 0) for t in recent_trades]
                avg_trade_return = np.clip(np.mean(trade_returns), -0.5, 0.5)
                trade_win_rate = np.mean([1 if r > 0 else 0 for r in trade_returns])
            else:
                avg_trade_return = 0
                trade_win_rate = 0.5
            
            # Assemble state vector
            state_components = [
                price_returns[-15:] if len(price_returns) >= 15 else 
                    np.pad(price_returns, (15 - len(price_returns), 0), 'constant'),
                technical_features,
                microstructure_features,
                alt_data_features,
                mtf_features,
                [
                    position_ratio, cash_ratio, total_return, current_drawdown,
                    volatility, trend, momentum, avg_trade_return, trade_win_rate,
                    len(self.trades) / 1000,  # Normalized trade count
                    self.current_step / len(self.market_data)  # Progress
                ]
            ]
            
            # Flatten and ensure correct size
            state = np.concatenate([np.array(comp).flatten() for comp in state_components])
            
            # [ERROR-HANDLING] Pad or truncate to exact size
            if len(state) < self.state_size:
                state = np.pad(state, (0, self.state_size - len(state)), 'constant')
            else:
                state = state[:self.state_size]
            
            # Final validation
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating state: {e}")
            logger.error(traceback.format_exc())
            self.state_errors += 1
            return np.zeros(self.state_size, dtype=np.float32)
    
    def _execute_action_safe(self, action: int) -> float:
        """
        Execute trading action with comprehensive error handling.
        
        Args:
            action: Integer action to execute
            
        Returns:
            Reward value
        """
        try:
            current_price = self.market_data['close'].iloc[self.current_step]
            
            # [ERROR-HANDLING] Validate price
            if current_price <= 0 or not np.isfinite(current_price):
                logger.error(f"Invalid price: {current_price}")
                return -1.0
            
            # Calculate trade size based on action
            trade_size = self._calculate_trade_size(action, current_price)
            
            if trade_size == 0:
                # Hold action
                return self._calculate_hold_reward()
            
            # Execute trade with validation
            executed_size = 0
            transaction_cost = 0
            
            if trade_size > 0:  # Buy
                executed_size = self._execute_buy(trade_size, current_price)
            else:  # Sell
                executed_size = self._execute_sell(abs(trade_size), current_price)
            
            # Calculate transaction cost
            if executed_size != 0:
                transaction_cost = abs(executed_size) * current_price * self.transaction_cost
            
            # Calculate reward
            reward = self._calculate_sophisticated_reward(executed_size, transaction_cost)
            
            # Update peak value and drawdown
            current_value = self.get_portfolio_value()
            if current_value > self.peak_value:
                self.peak_value = current_value
            
            return reward
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return -1.0
    
    def _calculate_trade_size(self, action: int, current_price: float) -> float:
        """Calculate trade size with risk limits"""
        try:
            # Action mapping
            if action == 0:  # Hold
                return 0
            
            # Position limits
            max_position_value = self.initial_balance * self.max_position_size
            current_position_value = self.position * current_price
            
            if action in [1, 2, 3]:  # Buy actions
                # Calculate maximum buy size
                max_buy_value = max_position_value - current_position_value
                max_buy_value = max(0, max_buy_value)
                
                # Available capital
                available_capital = self.balance
                
                # Action-specific sizing
                if action == 1:  # Buy 25%
                    buy_value = available_capital * 0.25
                elif action == 2:  # Buy 50%
                    buy_value = available_capital * 0.50
                else:  # Buy 100%
                    buy_value = available_capital
                
                # Apply limits
                buy_value = min(buy_value, max_buy_value, available_capital)
                trade_size = buy_value / current_price if current_price > 0 else 0
                
            else:  # Sell actions (4, 5, 6)
                if self.position <= 0:
                    return 0
                
                if action == 4:  # Sell 25%
                    trade_size = -self.position * 0.25
                elif action == 5:  # Sell 50%
                    trade_size = -self.position * 0.50
                else:  # Sell 100%
                    trade_size = -self.position
            
            # [ERROR-HANDLING] Validate trade size
            if not np.isfinite(trade_size):
                logger.warning(f"Invalid trade size calculated: {trade_size}")
                return 0
            
            return trade_size
            
        except Exception as e:
            logger.error(f"Error calculating trade size: {e}")
            return 0
    
    def _execute_buy(self, size: float, price: float) -> float:
        """Execute buy order with validation"""
        try:
            if size <= 0:
                return 0
            
            cost = size * price
            total_cost = cost * (1 + self.transaction_cost)
            
            # Check if we have enough balance
            if total_cost > self.balance:
                # Adjust size to available balance
                available_for_trade = self.balance / (1 + self.transaction_cost)
                size = available_for_trade / price
                cost = size * price
                total_cost = cost * (1 + self.transaction_cost)
            
            if size > 0.001:  # Minimum trade size
                self.balance -= total_cost
                self.position += size
                
                self.trades.append({
                    'type': 'buy',
                    'size': size,
                    'price': price,
                    'cost': cost * self.transaction_cost,
                    'timestamp': self.current_step,
                    'balance_after': self.balance,
                    'position_after': self.position
                })
                
                return size
            
            return 0
            
        except Exception as e:
            logger.error(f"Error executing buy: {e}")
            return 0
    
    def _execute_sell(self, size: float, price: float) -> float:
        """Execute sell order with validation"""
        try:
            if size <= 0 or self.position <= 0:
                return 0
            
            # Limit to available position
            sell_amount = min(size, self.position)
            
            if sell_amount > 0.001:  # Minimum trade size
                revenue = sell_amount * price
                transaction_cost = revenue * self.transaction_cost
                net_revenue = revenue - transaction_cost
                
                self.balance += net_revenue
                self.position -= sell_amount
                
                # Calculate return for this trade
                avg_buy_price = self._get_average_entry_price()
                trade_return = (price - avg_buy_price) / avg_buy_price if avg_buy_price > 0 else 0
                
                self.trades.append({
                    'type': 'sell',
                    'size': sell_amount,
                    'price': price,
                    'cost': transaction_cost,
                    'timestamp': self.current_step,
                    'balance_after': self.balance,
                    'position_after': self.position,
                    'return': trade_return
                })
                
                return -sell_amount  # Negative for sell
            
            return 0
            
        except Exception as e:
            logger.error(f"Error executing sell: {e}")
            return 0
    
    def _get_average_entry_price(self) -> float:
        """Calculate average entry price for current position"""
        try:
            if not self.trades:
                return 0
            
            total_bought = 0
            total_cost = 0
            
            for trade in self.trades:
                if trade['type'] == 'buy':
                    total_bought += trade['size']
                    total_cost += trade['size'] * trade['price']
            
            if total_bought > 0:
                return total_cost / total_bought
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating average entry price: {e}")
            return 0
    
    def _calculate_hold_reward(self) -> float:
        """Calculate reward for holding position"""
        if self.position == 0:
            return 0  # Neutral reward for no position
        
        # Small penalty to encourage active trading when appropriate
        return -0.01
    
    def _calculate_sophisticated_reward(self, executed_size: float, 
                                      transaction_cost: float) -> float:
        """
        Calculate sophisticated reward function with error handling.
        
        Args:
            executed_size: Size of executed trade
            transaction_cost: Cost of transaction
            
        Returns:
            Calculated reward value
        """
        try:
            current_value = self.get_portfolio_value()
            
            # Base portfolio return component
            portfolio_return = (current_value - self.initial_balance) / self.initial_balance
            return_component = portfolio_return * 100  # Scale up
            
            # [ERROR-HANDLING] Cap extreme returns
            return_component = np.clip(return_component, -100, 1000)
            
            # Risk-adjusted return (Sharpe ratio component)
            sharpe_component = 0
            if len(self.trades) >= 20:
                trade_returns = self._calculate_recent_trade_returns()
                if len(trade_returns) > 1:
                    mean_return = np.mean(trade_returns)
                    volatility = np.std(trade_returns)
                    if volatility > 1e-8:
                        sharpe_component = (mean_return / volatility) * 10
                        sharpe_component = np.clip(sharpe_component, -50, 50)
            
            # Drawdown penalty
            current_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
            drawdown_penalty = 0
            if current_drawdown > 0.05:  # 5% threshold
                drawdown_penalty = -50 * min((current_drawdown - 0.05) ** 2, 1)
            
            # Transaction cost penalty
            if self.initial_balance > 0:
                cost_penalty = -transaction_cost / self.initial_balance * 1000
                cost_penalty = max(cost_penalty, -10)  # Cap penalty
            else:
                cost_penalty = 0
            
            # Position management reward
            position_reward = self._calculate_position_reward()
            
            # Trading frequency penalty
            frequency_penalty = self._calculate_frequency_penalty()
            
            # Momentum alignment reward
            momentum_reward = self._calculate_momentum_reward(executed_size)
            
            # Combine all components with safety checks
            total_reward = (
                return_component + 
                sharpe_component + 
                drawdown_penalty + 
                cost_penalty + 
                position_reward + 
                frequency_penalty + 
                momentum_reward
            )
            
            # [ERROR-HANDLING] Final bounds check
            total_reward = np.clip(total_reward, -1000, 1000)
            
            # Check for NaN
            if not np.isfinite(total_reward):
                logger.warning("Non-finite reward calculated")
                return 0
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return -1.0
    
    def _calculate_position_reward(self) -> float:
        """Calculate position management reward"""
        try:
            position_value = self.position * self.market_data['close'].iloc[self.current_step]
            portfolio_value = self.get_portfolio_value()
            
            if portfolio_value > 0:
                position_ratio = position_value / portfolio_value
                
                # Reward balanced positions
                if 0.3 <= position_ratio <= 0.8:
                    return 2
                elif position_ratio > 0.95 or position_ratio < 0.05:
                    return -5
                else:
                    return 0
            
            return -10
            
        except Exception as e:
            logger.error(f"Error calculating position reward: {e}")
            return 0
    
    def _calculate_frequency_penalty(self) -> float:
        """Calculate trading frequency penalty"""
        try:
            recent_trades = len([
                t for t in self.trades 
                if self.current_step - t['timestamp'] <= 20
            ])
            
            if recent_trades > 10:
                return -2 * min(recent_trades - 10, 10)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating frequency penalty: {e}")
            return 0
    
    def _calculate_momentum_reward(self, executed_size: float) -> float:
        """Calculate momentum alignment reward"""
        try:
            if len(self.trades) == 0 or executed_size == 0:
                return 0
            
            last_trade = self.trades[-1]
            current_price = self.market_data['close'].iloc[self.current_step]
            
            if last_trade['type'] == 'buy':
                if current_price > last_trade['price']:
                    return 1  # Bought before price increase
                else:
                    return -0.5
            elif last_trade['type'] == 'sell':
                if current_price < last_trade['price']:
                    return 1  # Sold before price decrease
                else:
                    return -0.5
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating momentum reward: {e}")
            return 0
    
    def _calculate_recent_trade_returns(self, window: int = 20) -> List[float]:
        """Calculate returns for recent trades with error handling"""
        try:
            if len(self.trades) < 2:
                return []
            
            returns = []
            recent_trades = self.trades[-window:]
            
            for i in range(1, len(recent_trades)):
                prev_trade = recent_trades[i-1]
                curr_trade = recent_trades[i]
                
                if prev_trade['type'] == 'buy' and curr_trade['type'] == 'sell':
                    if prev_trade['price'] > 0:
                        trade_return = (curr_trade['price'] - prev_trade['price']) / prev_trade['price']
                        # Cap extreme returns
                        trade_return = np.clip(trade_return, -0.5, 0.5)
                        returns.append(trade_return)
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating trade returns: {e}")
            return []
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value with validation"""
        try:
            if self.current_step < len(self.market_data):
                current_price = self.market_data['close'].iloc[self.current_step]
            else:
                current_price = self.market_data['close'].iloc[-1]
            
            # Validate price
            if current_price <= 0 or not np.isfinite(current_price):
                logger.warning(f"Invalid price for portfolio value: {current_price}")
                current_price = self.market_data['close'].iloc[-1]
            
            value = self.balance + self.position * current_price
            
            # Validate result
            if not np.isfinite(value) or value < 0:
                logger.error(f"Invalid portfolio value: {value}")
                return self.balance  # Return just cash balance as fallback
            
            return value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return self.balance
    
    def _get_info(self) -> Dict:
        """Get additional information with error handling"""
        try:
            info = {
                'portfolio_value': self.get_portfolio_value(),
                'position': self.position,
                'balance': self.balance,
                'num_trades': len(self.trades),
                'max_drawdown': self.max_drawdown,
                'current_step': self.current_step,
                'episode': self.episode_count,
                'state_errors': self.state_errors
            }
            
            # Add performance metrics if available
            if self.trades:
                winning_trades = sum(1 for t in self.trades if t.get('return', 0) > 0)
                info['win_rate'] = winning_trades / len(self.trades)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting info: {e}")
            return {'error': str(e)}