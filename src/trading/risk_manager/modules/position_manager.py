"""
Position management module for trading risk management.
Handles position tracking, updates, and P&L calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import traceback

from utils.logger import setup_logger

logger = setup_logger(__name__)


class PositionManager:
    """Manages trading positions and their lifecycle"""
    
    def __init__(self, risk_params: Dict[str, float]):
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.risk_params = risk_params
        self.daily_trades = 0
        
    def validate_trade_inputs(self, symbol: str, side: str, size: float, price: float) -> bool:
        """Validate trade input parameters"""
        try:
            # Symbol validation
            if not symbol or not isinstance(symbol, str):
                logger.error(f"Invalid symbol: {symbol}")
                return False
            
            # Side validation
            if side not in ['long', 'short', 'buy', 'sell']:
                logger.error(f"Invalid side: {side}")
                return False
            
            # Size validation
            if not isinstance(size, (int, float)) or size <= 0 or not np.isfinite(size):
                logger.error(f"Invalid size: {size}")
                return False
            
            # Price validation
            if not isinstance(price, (int, float)) or price <= 0 or not np.isfinite(price):
                logger.error(f"Invalid price: {price}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade inputs: {e}")
            return False
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float, 
                    stop_loss: Optional[float] = None, current_capital: float = None) -> bool:
        """
        Add new position to portfolio with error handling.
        
        Args:
            symbol: Trading symbol
            side: Position side (long/short)
            size: Position size
            entry_price: Entry price
            stop_loss: Optional stop loss price
            current_capital: Current capital for calculations
            
        Returns:
            Success status
        """
        try:
            # Validate inputs
            if not self.validate_trade_inputs(symbol, side, size, entry_price):
                logger.error("Invalid position parameters")
                return False
            
            # Normalize side
            side = 'long' if side in ['long', 'buy'] else 'short'
            
            # Calculate stop loss if not provided
            if stop_loss is None:
                if side == 'long':
                    stop_loss = entry_price * (1 - self.risk_params['stop_loss_pct'])
                else:
                    stop_loss = entry_price * (1 + self.risk_params['stop_loss_pct'])
            
            # Validate stop loss
            if side == 'long' and stop_loss >= entry_price:
                logger.error(f"Invalid stop loss for long position: {stop_loss} >= {entry_price}")
                stop_loss = entry_price * 0.98
            elif side == 'short' and stop_loss <= entry_price:
                logger.error(f"Invalid stop loss for short position: {stop_loss} <= {entry_price}")
                stop_loss = entry_price * 1.02
            
            position = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'entry_time': pd.Timestamp.now(),
                'stop_loss': stop_loss,
                'highest_price': entry_price,
                'lowest_price': entry_price,
                'unrealized_pnl': 0,
                'realized_pnl': 0
            }
            
            if symbol in self.positions:
                # Update existing position
                self._update_position(symbol, position)
            else:
                self.positions[symbol] = position
            
            # Update trade history
            self.trade_history.append({
                'timestamp': position['entry_time'],
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': entry_price,
                'action': 'open',
                'capital_at_trade': current_capital
            })
            
            self.daily_trades += 1
            logger.info(f"Added position: {symbol} {side} {size:.4f} @ ${entry_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def update_position_price(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        Update position with current price and check stop loss.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Update result with stop loss trigger info
        """
        try:
            if symbol not in self.positions:
                return {'success': False, 'reason': 'Position not found'}
            
            # Validate price
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                logger.error(f"Invalid current price: {current_price}")
                return {'success': False, 'reason': 'Invalid price'}
            
            position = self.positions[symbol]
            
            # Update highest/lowest
            position['highest_price'] = max(position['highest_price'], current_price)
            position['lowest_price'] = min(position['lowest_price'], current_price)
            
            # Calculate unrealized PnL
            old_pnl = position['unrealized_pnl']
            
            if position['side'] == 'long':
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['size']
            else:
                position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['size']
            
            # Validate PnL
            if not np.isfinite(position['unrealized_pnl']):
                logger.error(f"Invalid PnL calculated for {symbol}")
                position['unrealized_pnl'] = old_pnl
                return {'success': False, 'reason': 'Invalid PnL calculation'}
            
            # Check stop loss
            stop_triggered = self._check_stop_loss(position, current_price)
            
            # Update trailing stop
            self._update_trailing_stop(position, current_price)
            
            return {
                'success': True,
                'stop_triggered': stop_triggered,
                'unrealized_pnl': position['unrealized_pnl'],
                'position': position
            }
            
        except Exception as e:
            logger.error(f"Error updating position price for {symbol}: {e}")
            return {'success': False, 'reason': str(e)}
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual") -> Dict[str, Any]:
        """
        Close position and record PnL with error handling.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Close result with PnL info
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"Position {symbol} not found")
                return {'success': False, 'reason': 'Position not found'}
            
            # Validate price
            if not isinstance(exit_price, (int, float)) or exit_price <= 0:
                logger.error(f"Invalid exit price: {exit_price}")
                return {'success': False, 'reason': 'Invalid exit price'}
            
            position = self.positions[symbol]
            
            # Calculate realized PnL
            if position['side'] == 'long':
                realized_pnl = (exit_price - position['entry_price']) * position['size']
            else:
                realized_pnl = (position['entry_price'] - exit_price) * position['size']
            
            # Validate PnL
            if not np.isfinite(realized_pnl):
                logger.error(f"Invalid PnL calculated: {realized_pnl}")
                realized_pnl = 0
            
            position['realized_pnl'] = realized_pnl
            
            # Update trade history
            self.trade_history.append({
                'timestamp': pd.Timestamp.now(),
                'symbol': symbol,
                'side': 'sell' if position['side'] == 'long' else 'buy',
                'size': position['size'],
                'price': exit_price,
                'action': 'close',
                'pnl': realized_pnl,
                'reason': reason
            })
            
            # Remove position
            closed_position = self.positions.pop(symbol)
            
            logger.info(
                f"Closed position: {symbol} @ ${exit_price:.2f}, "
                f"PnL: ${realized_pnl:.2f}, Reason: {reason}"
            )
            
            return {
                'success': True,
                'realized_pnl': realized_pnl,
                'closed_position': closed_position
            }
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            logger.error(traceback.format_exc())
            return {'success': False, 'reason': str(e)}
    
    def _check_stop_loss(self, position: Dict[str, Any], current_price: float) -> bool:
        """Check if stop loss is triggered with validation"""
        try:
            if 'stop_loss' not in position or position['stop_loss'] is None:
                return False
            
            stop_loss = position['stop_loss']
            
            # Validate stop loss value
            if not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
                logger.warning(f"Invalid stop loss: {stop_loss}")
                return False
            
            if position['side'] == 'long':
                return current_price <= stop_loss
            else:
                return current_price >= stop_loss
                
        except Exception as e:
            logger.error(f"Error checking stop loss: {e}")
            return False
    
    def _update_trailing_stop(self, position: Dict[str, Any], current_price: float) -> None:
        """Update trailing stop loss with error handling"""
        try:
            if position['side'] == 'long':
                # For long positions, trail stop up
                trailing_distance = position['highest_price'] * self.risk_params['stop_loss_pct']
                new_stop = position['highest_price'] - trailing_distance
                
                # Only update if new stop is higher
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
                    logger.debug(f"Updated trailing stop for {position['symbol']}: ${new_stop:.2f}")
                    
            else:
                # For short positions, trail stop down
                trailing_distance = position['lowest_price'] * self.risk_params['stop_loss_pct']
                new_stop = position['lowest_price'] + trailing_distance
                
                # Only update if new stop is lower
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
                    logger.debug(f"Updated trailing stop for {position['symbol']}: ${new_stop:.2f}")
                    
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
    
    def _update_position(self, symbol: str, new_position: Dict[str, Any]) -> None:
        """Update existing position with new trade"""
        try:
            existing = self.positions[symbol]
            
            # Same side - add to position
            if existing['side'] == new_position['side']:
                total_size = existing['size'] + new_position['size']
                
                # Weighted average price
                if total_size > 0:
                    avg_price = (
                        existing['entry_price'] * existing['size'] + 
                        new_position['entry_price'] * new_position['size']
                    ) / total_size
                else:
                    avg_price = existing['entry_price']
                
                existing['size'] = total_size
                existing['entry_price'] = avg_price
                existing['stop_loss'] = new_position['stop_loss']
                
            else:
                # Opposite side - reduce or flip position
                if new_position['size'] > existing['size']:
                    # Flip position
                    remaining_size = new_position['size'] - existing['size']
                    
                    # Calculate PnL on closed portion
                    if existing['side'] == 'long':
                        closed_pnl = (new_position['entry_price'] - existing['entry_price']) * existing['size']
                    else:
                        closed_pnl = (existing['entry_price'] - new_position['entry_price']) * existing['size']
                    
                    # Update to new position
                    existing.update({
                        'side': new_position['side'],
                        'size': remaining_size,
                        'entry_price': new_position['entry_price'],
                        'stop_loss': new_position['stop_loss'],
                        'highest_price': new_position['entry_price'],
                        'lowest_price': new_position['entry_price'],
                        'realized_pnl': closed_pnl
                    })
                else:
                    # Reduce position
                    existing['size'] -= new_position['size']
                    
                    # Calculate PnL on closed portion
                    if existing['side'] == 'long':
                        closed_pnl = (new_position['entry_price'] - existing['entry_price']) * new_position['size']
                    else:
                        closed_pnl = (existing['entry_price'] - new_position['entry_price']) * new_position['size']
                    
                    existing['realized_pnl'] = closed_pnl
                    
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure with error handling"""
        try:
            total = 0
            
            for position in self.positions.values():
                # Use current market price if available
                price = position.get('current_price', position['entry_price'])
                
                # Validate price
                if not isinstance(price, (int, float)) or price <= 0:
                    logger.warning(f"Invalid price for exposure calculation: {price}")
                    price = position['entry_price']
                
                exposure = position['size'] * price
                
                # Validate exposure
                if np.isfinite(exposure) and exposure >= 0:
                    total += exposure
                else:
                    logger.warning(f"Invalid exposure calculated: {exposure}")
            
            return total
            
        except Exception as e:
            logger.error(f"Error calculating total exposure: {e}")
            return 0
    
    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)
    
    def get_largest_position(self) -> float:
        """Get size of largest position"""
        if not self.positions:
            return 0
            
        try:
            return max(
                pos['size'] * pos.get('current_price', pos['entry_price'])
                for pos in self.positions.values()
            )
        except Exception:
            return 0
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL"""
        return sum(
            pos.get('unrealized_pnl', 0) 
            for pos in self.positions.values()
            if np.isfinite(pos.get('unrealized_pnl', 0))
        )
    
    def reset_daily_counters(self) -> None:
        """Reset daily trade counter"""
        self.daily_trades = 0
