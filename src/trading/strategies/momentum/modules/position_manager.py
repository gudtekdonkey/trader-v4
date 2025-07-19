"""
Momentum Strategy - Position Manager Module
Handles position tracking, updates, and exit logic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import traceback
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class PositionManager:
    """Manage momentum positions and exit conditions"""
    
    def __init__(self, params: Dict):
        """
        Initialize position manager
        
        Args:
            params: Strategy parameters
        """
        self.params = params
        self.active_positions = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        self.errors = 0
    
    def add_position(self, symbol: str, position_data: Dict):
        """Add a new position to track"""
        self.active_positions[symbol] = {
            **position_data,
            'entry_time': pd.Timestamp.now()
        }
        logger.info(f"Added momentum position for {symbol}: {position_data}")
    
    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Update positions with comprehensive error handling"""
        actions = []
        
        try:
            for symbol, position in list(self.active_positions.items()):
                try:
                    if symbol not in current_prices:
                        logger.warning(f"No price data for {symbol}")
                        continue
                    
                    current_price = current_prices[symbol]
                    if not self._is_valid_price(current_price):
                        logger.error(f"Invalid price for {symbol}: {current_price}")
                        continue
                    
                    # Calculate unrealized PnL
                    if position['direction'] == 1:  # Long
                        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    else:  # Short
                        pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    
                    # Validate PnL
                    if not np.isfinite(pnl_pct):
                        logger.error(f"Invalid PnL calculation for {symbol}")
                        continue
                    
                    # Update position info
                    position['current_price'] = current_price
                    position['unrealized_pnl'] = pnl_pct
                    
                    # Check for stop loss hit
                    if self._check_stop_loss(position, current_price):
                        actions.append({
                            'action': 'close_position',
                            'symbol': symbol,
                            'reason': 'stop_loss_hit',
                            'price': current_price
                        })
                        self._close_position(symbol, current_price, 'stop_loss')
                        continue
                    
                    # Check for take profit hit
                    if self._check_take_profit(position, current_price):
                        actions.append({
                            'action': 'close_position',
                            'symbol': symbol,
                            'reason': 'take_profit_hit',
                            'price': current_price
                        })
                        self._close_position(symbol, current_price, 'take_profit')
                        continue
                    
                    # Trailing stop loss with validation
                    if pnl_pct > 0.02:  # 2% profit
                        try:
                            new_stop = self._calculate_trailing_stop(
                                position, current_price, pnl_pct
                            )
                            if new_stop != position['stop_loss']:
                                actions.append({
                                    'action': 'update_stop_loss',
                                    'symbol': symbol,
                                    'new_stop': new_stop,
                                    'old_stop': position['stop_loss']
                                })
                                position['stop_loss'] = new_stop
                        except Exception as e:
                            logger.error(f"Error calculating trailing stop: {e}")
                    
                    # Check for manual exit conditions
                    if self._should_exit_position(position, current_price, pnl_pct):
                        actions.append({
                            'action': 'close_position',
                            'symbol': symbol,
                            'reason': 'momentum_exhausted',
                            'price': current_price
                        })
                        self._close_position(symbol, current_price, 'momentum_exhausted')
                        
                except Exception as e:
                    logger.error(f"Error updating position {symbol}: {e}")
                    self.errors += 1
                    continue
            
            return actions
            
        except Exception as e:
            logger.error(f"Critical error in update_positions: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _is_valid_price(self, price: float) -> bool:
        """Check if price is valid"""
        return isinstance(price, (int, float)) and price > 0 and np.isfinite(price)
    
    def _check_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if position['direction'] == 1:  # Long
            return current_price <= position['stop_loss']
        else:  # Short
            return current_price >= position['stop_loss']
    
    def _check_take_profit(self, position: Dict, current_price: float) -> bool:
        """Check if take profit is hit"""
        if position['direction'] == 1:  # Long
            return current_price >= position['take_profit']
        else:  # Short
            return current_price <= position['take_profit']
    
    def _calculate_trailing_stop(self, position: Dict, current_price: float, pnl_pct: float) -> float:
        """Calculate trailing stop with validation"""
        try:
            if position['direction'] == 1:  # Long
                # Move stop to breakeven at 2% profit
                if pnl_pct >= 0.02 and position['stop_loss'] < position['entry_price']:
                    new_stop = position['entry_price'] * 1.001  # Small buffer above entry
                elif pnl_pct >= 0.05:  # Trail at 50% of profit above 5%
                    trail_distance = (current_price - position['entry_price']) * 0.5
                    new_stop = position['entry_price'] + trail_distance
                    new_stop = max(new_stop, position['stop_loss'])
                else:
                    new_stop = position['stop_loss']
            else:  # Short
                # Move stop to breakeven at 2% profit
                if pnl_pct >= 0.02 and position['stop_loss'] > position['entry_price']:
                    new_stop = position['entry_price'] * 0.999  # Small buffer below entry
                elif pnl_pct >= 0.05:  # Trail at 50% of profit above 5%
                    trail_distance = (position['entry_price'] - current_price) * 0.5
                    new_stop = position['entry_price'] - trail_distance
                    new_stop = min(new_stop, position['stop_loss'])
                else:
                    new_stop = position['stop_loss']
            
            # Validate new stop
            if position['direction'] == 1:
                if new_stop >= current_price:
                    logger.warning("Invalid trailing stop for long position")
                    return position['stop_loss']
            else:
                if new_stop <= current_price:
                    logger.warning("Invalid trailing stop for short position")
                    return position['stop_loss']
            
            return new_stop
            
        except Exception as e:
            logger.error(f"Error in trailing stop calculation: {e}")
            return position['stop_loss']
    
    def _should_exit_position(self, position: Dict, current_price: float, pnl_pct: float) -> bool:
        """Check exit conditions with error handling"""
        try:
            # Time-based exit
            if 'entry_time' in position:
                time_in_position = pd.Timestamp.now() - position['entry_time']
                if time_in_position > pd.Timedelta(hours=24):
                    return True
            
            # Loss threshold
            if pnl_pct < -0.05:  # 5% loss
                return True
            
            # Profit taking at extended levels
            if pnl_pct > 0.10:  # 10% profit
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close position and update performance"""
        try:
            position = self.active_positions[symbol]
            
            # Calculate final PnL
            if position['direction'] == 1:  # Long
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            else:  # Short
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
            
            # Update performance
            self._update_performance(pnl_pct)
            
            # Remove position
            del self.active_positions[symbol]
            
            logger.info(f"Closed position {symbol}: reason={reason}, PnL={pnl_pct:.2%}")
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
    
    def _update_performance(self, pnl_pct: float):
        """Update strategy performance metrics"""
        try:
            self.performance['total_trades'] += 1
            self.performance['total_pnl'] += pnl_pct
            
            if pnl_pct > 0:
                self.performance['winning_trades'] += 1
            
            # Update max drawdown
            if pnl_pct < 0:
                self.performance['max_drawdown'] = min(
                    self.performance['max_drawdown'], 
                    pnl_pct
                )
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_active_positions(self) -> Dict:
        """Get all active positions"""
        return self.active_positions.copy()
    
    def get_position_count(self) -> int:
        """Get number of active positions"""
        return len(self.active_positions)
    
    def has_position(self, symbol: str) -> bool:
        """Check if symbol has active position"""
        return symbol in self.active_positions
    
    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        try:
            # Calculate win rate
            win_rate = (
                self.performance['winning_trades'] / self.performance['total_trades']
                if self.performance['total_trades'] > 0 else 0
            )
            
            # Calculate average trade
            avg_trade = (
                self.performance['total_pnl'] / self.performance['total_trades']
                if self.performance['total_trades'] > 0 else 0
            )
            
            # Calculate Sharpe ratio (simplified)
            if self.performance['total_trades'] > 1:
                returns = []
                for symbol, pos in self.active_positions.items():
                    if 'unrealized_pnl' in pos:
                        returns.append(pos['unrealized_pnl'])
                
                if returns:
                    returns_std = np.std(returns)
                    if returns_std > 0:
                        sharpe = (avg_trade / returns_std) * np.sqrt(252)  # Annualized
                    else:
                        sharpe = 0
                else:
                    sharpe = 0
            else:
                sharpe = 0
            
            return {
                **self.performance,
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'sharpe_ratio': sharpe,
                'active_positions': len(self.active_positions),
                'position_errors': self.errors
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self.performance
