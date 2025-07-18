"""
Reward functions for different trading strategies
"""

import numpy as np
from typing import TYPE_CHECKING, Optional
from ...utils.logger import setup_logger

if TYPE_CHECKING:
    from .trading_environment import CryptoTradingEnvironment

logger = setup_logger(__name__)


class RewardFunctions:
    """Collection of reward functions for different trading strategies"""
    
    @staticmethod
    def trend_following_reward(env: 'CryptoTradingEnvironment', 
                             executed_size: float, 
                             transaction_cost: float) -> float:
        """Trend following reward with error handling"""
        try:
            current_step = env.current_step
            market_data = env.market_data
            
            # Calculate trend indicators safely
            lookback = 20
            if current_step >= lookback:
                prices = market_data['close'].iloc[current_step-lookback:current_step+1].values
                
                # Validate prices
                if len(prices) > 0 and np.all(prices > 0):
                    # Linear regression for trend
                    x = np.arange(len(prices))
                    slope = np.polyfit(x, prices, 1)[0]
                    
                    price_range = prices.max() - prices.min()
                    if price_range > 0:
                        trend_strength = np.clip(slope / price_range * lookback, -1, 1)
                    else:
                        trend_strength = 0
                else:
                    trend_strength = 0
                    
                # Moving average signal
                if current_step >= 50:
                    ma_fast = market_data['close'].iloc[current_step-10:current_step].mean()
                    ma_slow = market_data['close'].iloc[current_step-50:current_step].mean()
                    ma_signal = 1 if ma_fast > ma_slow else -1
                else:
                    ma_signal = 0
            else:
                trend_strength = 0
                ma_signal = 0
            
            # Base reward
            current_value = env.get_portfolio_value()
            portfolio_return = (current_value - env.initial_balance) / env.initial_balance
            
            # Position alignment
            position_value = env.position * market_data['close'].iloc[current_step]
            portfolio_value = max(current_value, 1)  # Avoid division by zero
            position_ratio = position_value / portfolio_value
            
            # Trend alignment reward
            trend_alignment_reward = 0
            if trend_strength > 0.2:  # Uptrend
                if executed_size > 0:
                    trend_alignment_reward = 10 * trend_strength * abs(executed_size) / env.initial_balance
                elif executed_size < 0 and position_ratio < 0.1:
                    trend_alignment_reward = -5
            elif trend_strength < -0.2:  # Downtrend
                if executed_size < 0:
                    trend_alignment_reward = 10 * abs(trend_strength) * abs(executed_size) / env.initial_balance
                elif executed_size > 0:
                    trend_alignment_reward = -10 * abs(executed_size) / env.initial_balance
            else:
                trend_alignment_reward = -2 if abs(executed_size) > 0 else 0
            
            # Riding winners reward
            riding_winner_reward = 0
            if env.position > 0 and len(env.trades) > 0:
                last_buy_trade = None
                for trade in reversed(env.trades):
                    if trade['type'] == 'buy':
                        last_buy_trade = trade
                        break
                
                if last_buy_trade:
                    current_price = market_data['close'].iloc[current_step]
                    if last_buy_trade['price'] > 0:
                        unrealized_return = (current_price - last_buy_trade['price']) / last_buy_trade['price']
                        if unrealized_return > 0.02:
                            riding_winner_reward = min(unrealized_return * 50, 10)
            
            # Counter-trend penalty
            counter_trend_penalty = 0
            if position_ratio > 0.3:
                if trend_strength < -0.3 and ma_signal < 0:
                    counter_trend_penalty = -15 * position_ratio
                elif trend_strength > 0.3 and position_ratio < 0.1:
                    counter_trend_penalty = -5
            
            # Transaction cost
            cost_penalty = -transaction_cost / env.initial_balance * 500
            
            # Total reward with bounds
            total_reward = (
                portfolio_return * 100 +
                trend_alignment_reward +
                riding_winner_reward +
                counter_trend_penalty +
                cost_penalty
            )
            
            return np.clip(total_reward, -1000, 1000)
            
        except Exception as e:
            logger.error(f"Error in trend following reward: {e}")
            return -1.0
    
    @staticmethod
    def mean_reversion_reward(env: 'CryptoTradingEnvironment',
                            executed_size: float,
                            transaction_cost: float) -> float:
        """Mean reversion reward with error handling"""
        try:
            current_step = env.current_step
            market_data = env.market_data
            
            # Get indicators safely
            current_data = market_data.iloc[current_step]
            rsi = current_data.get('rsi_14', 50)
            bb_position = current_data.get('bb_position', 0.5)
            
            # Calculate z-score
            lookback = 20
            z_score = 0
            if current_step >= lookback:
                prices = market_data['close'].iloc[current_step-lookback:current_step+1]
                if len(prices) > 0:
                    price_mean = prices.mean()
                    price_std = prices.std()
                    current_price = current_data['close']
                    
                    if price_std > 0 and price_mean > 0:
                        z_score = (current_price - price_mean) / price_std
                        z_score = np.clip(z_score, -3, 3)
            
            # Base reward
            current_value = env.get_portfolio_value()
            portfolio_return = (current_value - env.initial_balance) / env.initial_balance
            
            # Mean reversion signals
            reversion_reward = 0
            
            if executed_size > 0:  # Buying
                if rsi < 30:
                    reversion_reward += 5 * (30 - rsi) / 30
                if bb_position < 0.2:
                    reversion_reward += 5 * (0.2 - bb_position) / 0.2
                if z_score < -2:
                    reversion_reward += 8
                elif z_score < -1:
                    reversion_reward += 4
                
                # Penalties
                if rsi > 70:
                    reversion_reward -= 8
                if z_score > 1:
                    reversion_reward -= 5
                    
            elif executed_size < 0:  # Selling
                if rsi > 70:
                    reversion_reward += 5 * (rsi - 70) / 30
                if bb_position > 0.8:
                    reversion_reward += 5 * (bb_position - 0.8) / 0.2
                if z_score > 2:
                    reversion_reward += 8
                elif z_score > 1:
                    reversion_reward += 4
                
                # Penalties
                if rsi < 30:
                    reversion_reward -= 8
                if z_score < -1:
                    reversion_reward -= 5
            
            # Profit taking reward
            profit_taking_reward = 0
            if len(env.trades) > 0:
                last_trade = env.trades[-1]
                if last_trade['type'] == 'buy' and executed_size < 0:
                    if last_trade['price'] > 0:
                        trade_return = (current_data['close'] - last_trade['price']) / last_trade['price']
                        if 0.005 < trade_return < 0.02:
                            profit_taking_reward = 10 * trade_return / 0.02
                        elif trade_return > 0.02:
                            profit_taking_reward = 5
            
            # Transaction cost
            cost_penalty = -transaction_cost / env.initial_balance * 800
            
            # Total reward
            total_reward = (
                portfolio_return * 100 +
                reversion_reward +
                profit_taking_reward +
                cost_penalty
            )
            
            return np.clip(total_reward, -1000, 1000)
            
        except Exception as e:
            logger.error(f"Error in mean reversion reward: {e}")
            return -1.0
    
    @staticmethod
    def volatility_trading_reward(env: 'CryptoTradingEnvironment',
                                executed_size: float,
                                transaction_cost: float) -> float:
        """Volatility trading reward with error handling"""
        try:
            current_step = env.current_step
            market_data = env.market_data
            
            # Calculate volatility metrics
            short_vol = 0.2
            vol_ratio = 1.0
            
            if current_step >= 30:
                recent_prices = market_data['close'].iloc[current_step-30:current_step+1]
                if len(recent_prices) > 1:
                    returns = np.diff(np.log(recent_prices + 1e-8))
                    
                    # Short-term volatility
                    if len(returns) >= 10:
                        short_vol = np.std(returns[-10:]) * np.sqrt(252)
                        short_vol = np.clip(short_vol, 0, 1)
                    
                    # Long-term volatility
                    long_vol = np.std(returns) * np.sqrt(252)
                    long_vol = max(long_vol, 0.01)
                    
                    vol_ratio = short_vol / long_vol
            
            # Get ATR
            current_data = market_data.iloc[current_step]
            atr = current_data.get('atr', 0)
            atr_percent = atr / current_data['close'] if current_data['close'] > 0 else 0.02
            
            # Base reward
            current_value = env.get_portfolio_value()
            portfolio_return = (current_value - env.initial_balance) / env.initial_balance
            
            # Volatility regime rewards
            volatility_reward = 0
            
            is_high_vol = short_vol > 0.3 or atr_percent > 0.03
            is_low_vol = short_vol < 0.15 and atr_percent < 0.01
            is_vol_expansion = vol_ratio > 1.5
            
            # High volatility trading
            if is_high_vol and abs(executed_size) > 0:
                volatility_reward += 8 * min(short_vol / 0.3, 2)
                
                # Position sizing reward
                ideal_position_size = 0.5 / (short_vol / 0.2)
                actual_position_ratio = abs(executed_size) * current_data['close'] / env.initial_balance
                
                if 0.8 * ideal_position_size <= actual_position_ratio <= 1.2 * ideal_position_size:
                    volatility_reward += 5
            
            # Low volatility penalty
            elif is_low_vol and abs(executed_size) > 0:
                volatility_reward -= 10
            
            # Volatility expansion reward
            if is_vol_expansion:
                if abs(executed_size) > 0:
                    volatility_reward += 6
                if env.position > 0:
                    volatility_reward += 4
            
            # Transaction cost
            cost_penalty = -transaction_cost / env.initial_balance * 600
            
            # Total reward
            total_reward = (
                portfolio_return * 100 +
                volatility_reward +
                cost_penalty
            )
            
            return np.clip(total_reward, -1000, 1000)
            
        except Exception as e:
            logger.error(f"Error in volatility trading reward: {e}")
            return -1.0
    
    @staticmethod
    def momentum_trading_reward(env: 'CryptoTradingEnvironment',
                              executed_size: float,
                              transaction_cost: float) -> float:
        """Momentum trading reward with error handling"""
        try:
            current_step = env.current_step
            market_data = env.market_data
            
            # Calculate momentum metrics
            momentum_metrics = {}
            
            if current_step >= 20:
                try:
                    # Rate of change
                    current_price = market_data.iloc[current_step]['close']
                    for period in [5, 10, 20]:
                        if current_step >= period:
                            past_price = market_data.iloc[current_step-period]['close']
                            if past_price > 0:
                                roc = (current_price - past_price) / past_price
                                momentum_metrics[f'roc_{period}'] = np.clip(roc, -0.5, 0.5)
                            else:
                                momentum_metrics[f'roc_{period}'] = 0
                    
                    # Momentum consistency
                    recent_returns = []
                    for i in range(1, min(11, current_step + 1)):
                        past_price = market_data.iloc[current_step-i]['close']
                        if past_price > 0:
                            ret = (market_data.iloc[current_step-i+1]['close'] - past_price) / past_price
                            recent_returns.append(ret)
                    
                    if recent_returns:
                        positive_days = sum(1 for r in recent_returns if r > 0)
                        momentum_metrics['consistency'] = positive_days / len(recent_returns)
                    else:
                        momentum_metrics['consistency'] = 0.5
                        
                except Exception as e:
                    logger.warning(f"Error calculating momentum metrics: {e}")
                    momentum_metrics = {'roc_5': 0, 'roc_10': 0, 'roc_20': 0, 'consistency': 0.5}
            else:
                momentum_metrics = {'roc_5': 0, 'roc_10': 0, 'roc_20': 0, 'consistency': 0.5}
            
            # Base reward
            current_value = env.get_portfolio_value()
            portfolio_return = (current_value - env.initial_balance) / env.initial_balance
            
            # Momentum entry rewards
            momentum_reward = 0
            
            if executed_size > 0:  # Buying
                if momentum_metrics.get('roc_5', 0) > 0.02 and momentum_metrics.get('roc_10', 0) > 0.03:
                    momentum_reward += 8 * min(momentum_metrics['roc_10'] / 0.05, 2)
                
                if momentum_metrics.get('consistency', 0.5) > 0.7:
                    momentum_reward += 5
                
                # Penalty for buying against momentum
                if momentum_metrics.get('roc_5', 0) < -0.02:
                    momentum_reward -= 10
                    
            elif executed_size < 0:  # Selling
                if env.position > 0:
                    # Reward profit taking on strength
                    if momentum_metrics.get('roc_10', 0) > 0.05:
                        momentum_reward += 4
                    
                    # Reward exiting on momentum loss
                    if momentum_metrics.get('roc_5', 0) < 0 and momentum_metrics.get('consistency', 0.5) < 0.3:
                        momentum_reward += 8
                
                # Shorting on negative momentum
                if momentum_metrics.get('roc_5', 0) < -0.02 and momentum_metrics.get('roc_10', 0) < -0.03:
                    momentum_reward += 6
            
            # Holding winners
            if env.position > 0 and executed_size == 0:
                if momentum_metrics.get('roc_5', 0) > 0.01 and momentum_metrics.get('consistency', 0.5) > 0.6:
                    momentum_reward += 5
            
            # Transaction cost
            cost_penalty = -transaction_cost / env.initial_balance * 600
            
            # Total reward
            total_reward = (
                portfolio_return * 100 +
                momentum_reward +
                cost_penalty
            )
            
            return np.clip(total_reward, -1000, 1000)
            
        except Exception as e:
            logger.error(f"Error in momentum trading reward: {e}")
            return -1.0