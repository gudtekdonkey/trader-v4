"""
Mean Reversion Strategy - Position Manager Module
Handles position tracking, updates, and exit logic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import traceback
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class PositionManager:
    """Manage mean reversion positions and exit conditions"""
    
    def __init__(self, params: Dict):
        """
        Initialize position manager
        
        Args:
            params: Strategy parameters
        """
        self.params = params
        self.active_positions = {}
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'win_rate': 0,
            'last_update': pd.Timestamp.now()
        }
        self.error_stats = {
            'position_update_errors': 0,
            'last_error': None
        }
    
    def add_position(self, symbol: str, position_data: Dict):
        """Add a new position to track"""
        self.active_positions[symbol] = {
            **position_data,
            'entry_time': pd.Timestamp.now(),
            'bars_held': 0
        }
        logger.info(f"Added position for {symbol}: {position_data}")
    
    def update_positions(self, current_data: Dict[str, pd.DataFrame], 
                        indicator_calculator) -> List[Dict]:
        """Update positions based on current market data"""
        actions = []
        
        try:
            for symbol, position in list(self.active_positions.items()):
                try:
                    if symbol not in current_data:
                        logger.warning(f"No data for symbol {symbol} in position update")
                        continue
                    
                    df = current_data[symbol]
                    if df.empty:
                        logger.warning(f"Empty DataFrame for {symbol}")
                        continue
                    
                    # Calculate indicators for current data
                    indicators = indicator_calculator.calculate_all(df)
                    if not indicators:
                        logger.warning(f"Failed to calculate indicators for {symbol}")
                        continue
                    
                    current_idx = -1
                    
                    # Get current values safely
                    try:
                        current_price = float(df['close'].iloc[current_idx])
                        current_z_score = float(indicators['z_score'].iloc[current_idx])
                        
                        if not np.isfinite(current_price) or not np.isfinite(current_z_score):
                            logger.warning(f"Invalid values for {symbol}: price={current_price}, z_score={current_z_score}")
                            continue
                    except Exception as e:
                        logger.error(f"Error getting current values for {symbol}: {e}")
                        continue
                    
                    # Update bars held
                    position['bars_held'] = position.get('bars_held', 0) + 1
                    
                    # Check exit conditions
                    should_exit, exit_reason = self._check_exit_conditions(
                        position, current_price, current_z_score, indicators, current_idx
                    )
                    
                    if should_exit:
                        actions.append({
                            'action': 'close_position',
                            'symbol': symbol,
                            'reason': exit_reason,
                            'current_price': current_price,
                            'position': position
                        })
                        
                        # Calculate and record PnL
                        try:
                            if position['direction'] == 1:
                                pnl = (current_price - position['entry_price']) / position['entry_price']
                            else:
                                pnl = (position['entry_price'] - current_price) / position['entry_price']
                            
                            self._update_stats(pnl)
                            
                            # Remove position
                            del self.active_positions[symbol]
                            logger.info(f"Closed position for {symbol}: {exit_reason}, PnL: {pnl:.2%}")
                        except Exception as e:
                            logger.error(f"Error calculating PnL for {symbol}: {e}")
                            
                except Exception as e:
                    logger.error(f"Error updating position for {symbol}: {e}")
                    self.error_stats['position_update_errors'] += 1
                    continue
                    
        except Exception as e:
            logger.error(f"Critical error in update_positions: {e}")
            logger.error(traceback.format_exc())
            self.error_stats['position_update_errors'] += 1
        
        return actions
    
    def _check_exit_conditions(self, position: Dict, current_price: float, current_z_score: float,
                              indicators: Dict[str, pd.Series], idx: int) -> Tuple[bool, str]:
        """Check if position should be exited with error handling"""
        try:
            # Validate inputs
            if not np.isfinite(current_price) or not np.isfinite(current_z_score):
                logger.warning("Invalid price or z-score in exit check")
                return True, 'invalid_data'
            
            # Stop loss hit
            if position['direction'] == 1:  # Long
                if current_price <= position.get('stop_loss', 0):
                    return True, 'stop_loss_hit'
            else:  # Short
                if current_price >= position.get('stop_loss', float('inf')):
                    return True, 'stop_loss_hit'
            
            # Take profit hit
            if position['direction'] == 1:  # Long
                if current_price >= position.get('take_profit', float('inf')):
                    return True, 'take_profit_hit'
            else:  # Short
                if current_price <= position.get('take_profit', 0):
                    return True, 'take_profit_hit'
            
            # Z-score crossed zero (mean reversion complete)
            if abs(current_z_score) <= self.params['exit_z_score']:
                return True, 'mean_reversion_complete'
            
            # Z-score reversed (beyond opposite threshold)
            entry_z_score = position.get('entry_z_score', 0)
            if entry_z_score > 0 and current_z_score < -self.params['exit_z_score']:
                return True, 'z_score_reversal'
            elif entry_z_score < 0 and current_z_score > self.params['exit_z_score']:
                return True, 'z_score_reversal'
            
            # Maximum holding period reached
            bars_held = position.get('bars_held', 0)
            if bars_held >= self.params['position_hold_bars']:
                return True, 'max_holding_period'
            
            # Volatility spike (market regime change)
            try:
                current_vol = float(indicators['volatility'].iloc[idx])
                if current_vol > self.params['max_volatility'] * 1.5:
                    return True, 'volatility_spike'
            except Exception:
                pass  # Don't exit on volatility check failure
            
            # Efficiency ratio spike (trending market)
            try:
                efficiency_ratio = float(indicators['efficiency_ratio'].iloc[idx])
                if efficiency_ratio > 0.8:  # Strong trend developing
                    return True, 'market_trending'
            except Exception:
                pass
            
            return False, ''
            
        except Exception as e:
            logger.error(f"Error in exit condition check: {e}")
            # Exit position on error
            return True, 'error_in_exit_check'
    
    def _update_stats(self, pnl: float):
        """Update strategy statistics with error handling"""
        try:
            # Validate PnL
            if not np.isfinite(pnl):
                logger.warning(f"Invalid PnL value: {pnl}")
                return
            
            self.stats['total_trades'] += 1
            
            if pnl > 0:
                self.stats['winning_trades'] += 1
                # Update average win
                if self.stats['winning_trades'] > 1:
                    self.stats['avg_win'] = (
                        (self.stats['avg_win'] * (self.stats['winning_trades'] - 1) + pnl) / 
                        self.stats['winning_trades']
                    )
                else:
                    self.stats['avg_win'] = pnl
            else:
                # Update average loss
                losing_trades = self.stats['total_trades'] - self.stats['winning_trades']
                if losing_trades > 1:
                    self.stats['avg_loss'] = (
                        (self.stats['avg_loss'] * (losing_trades - 1) + pnl) / 
                        losing_trades
                    )
                else:
                    self.stats['avg_loss'] = pnl
            
            self.stats['total_pnl'] += pnl
            self.stats['win_rate'] = (
                self.stats['winning_trades'] / self.stats['total_trades'] 
                if self.stats['total_trades'] > 0 else 0
            )
            self.stats['last_update'] = pd.Timestamp.now()
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def get_active_positions(self) -> Dict:
        """Get all active positions"""
        return self.active_positions.copy()
    
    def get_position_count(self) -> int:
        """Get number of active positions"""
        return len(self.active_positions)
    
    def has_position(self, symbol: str) -> bool:
        """Check if symbol has active position"""
        return symbol in self.active_positions
    
    def get_statistics(self) -> Dict:
        """Get position management statistics"""
        return {
            'stats': self.stats.copy(),
            'active_positions': len(self.active_positions),
            'error_stats': self.error_stats.copy()
        }
