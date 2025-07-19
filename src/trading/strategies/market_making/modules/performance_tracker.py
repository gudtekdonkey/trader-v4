"""
Performance tracking module for market making strategy.
Handles metrics calculation, PnL tracking, and strategy analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks and analyzes market making strategy performance"""
    
    def __init__(self):
        self.performance = {
            'total_trades': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'gross_pnl': 0,
            'fee_revenue': 0,
            'inventory_pnl': 0,
            'spread_captured': 0,
            'quote_failures': 0,
            'errors': 0
        }
        
        # Microstructure tracking
        self.microstructure = {}
        
        # Time-based metrics
        self.hourly_metrics = deque(maxlen=24)
        self.daily_metrics = deque(maxlen=30)
        
        # Trade statistics
        self.trade_stats = {
            'spreads_captured': deque(maxlen=1000),
            'fill_rates': deque(maxlen=100),
            'quote_lifetimes': deque(maxlen=1000)
        }
        
    def update_fill_metrics(self, fill_data: Dict):
        """Update metrics based on order fill"""
        try:
            symbol = fill_data['symbol']
            side = fill_data['side']
            size = float(fill_data['size'])
            price = float(fill_data['price'])
            fee = float(fill_data.get('fee', 0))
            
            # Update volume metrics
            if side == 'buy':
                self.performance['buy_volume'] += size * price
            else:
                self.performance['sell_volume'] += size * price
            
            # Update trade count and fees
            self.performance['total_trades'] += 1
            self.performance['fee_revenue'] += fee  # Negative for maker rebates
            
            logger.debug(f"Updated fill metrics: {symbol} {side} {size} @ {price}")
            
        except Exception as e:
            logger.error(f"Error updating fill metrics: {e}")
            self.performance['errors'] += 1
    
    def update_spread_captured(self, spread_value: float):
        """Update spread capture metrics"""
        try:
            if spread_value > 0:
                self.performance['spread_captured'] += spread_value
                self.trade_stats['spreads_captured'].append(spread_value)
                
        except Exception as e:
            logger.error(f"Error updating spread captured: {e}")
    
    def update_microstructure(self, symbol: str, market_data: Dict):
        """Update market microstructure tracking"""
        try:
            if symbol not in self.microstructure:
                self.microstructure[symbol] = {
                    'avg_spread': deque(maxlen=100),
                    'avg_volume': deque(maxlen=100),
                    'volatility': deque(maxlen=100),
                    'order_imbalance': deque(maxlen=100),
                    'last_update': pd.Timestamp.now()
                }
            
            ms = self.microstructure[symbol]
            
            # Update metrics with validation
            if 'best_ask' in market_data and 'best_bid' in market_data:
                spread = market_data['best_ask'] - market_data['best_bid']
                if spread > 0:
                    ms['avg_spread'].append(spread)
            
            if 'volume' in market_data and isinstance(market_data['volume'], (int, float)):
                if market_data['volume'] >= 0:
                    ms['avg_volume'].append(market_data['volume'])
            
            if 'volatility' in market_data and isinstance(market_data['volatility'], (int, float)):
                if 0 <= market_data['volatility'] <= 1:
                    ms['volatility'].append(market_data['volatility'])
            
            if 'order_imbalance' in market_data and isinstance(market_data['order_imbalance'], (int, float)):
                if -1 <= market_data['order_imbalance'] <= 1:
                    ms['order_imbalance'].append(market_data['order_imbalance'])
            
            ms['last_update'] = pd.Timestamp.now()
                    
        except Exception as e:
            logger.error(f"Error updating microstructure: {e}")
    
    def update_quote_metrics(self, quotes_placed: int, quotes_failed: int):
        """Update quote placement metrics"""
        try:
            if quotes_placed + quotes_failed > 0:
                fill_rate = quotes_placed / (quotes_placed + quotes_failed)
                self.trade_stats['fill_rates'].append(fill_rate)
            
            self.performance['quote_failures'] += quotes_failed
            
        except Exception as e:
            logger.error(f"Error updating quote metrics: {e}")
    
    def calculate_sharpe_ratio(self, returns: Optional[List[float]] = None) -> float:
        """Calculate Sharpe ratio from returns or spread capture"""
        try:
            if returns is None:
                # Use spread capture as proxy for returns
                if len(self.trade_stats['spreads_captured']) < 2:
                    return 0
                
                returns = list(self.trade_stats['spreads_captured'])
            
            if len(returns) < 2:
                return 0
            
            # Calculate Sharpe ratio (assuming 0 risk-free rate)
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
                return min(10, max(-10, sharpe))  # Cap at reasonable values
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def calculate_inventory_turnover(self, inventory_metrics: Dict) -> float:
        """Calculate inventory turnover rate"""
        try:
            total_volume = self.performance['buy_volume'] + self.performance['sell_volume']
            gross_exposure = inventory_metrics.get('gross_exposure', 0)
            
            if gross_exposure > 0 and total_volume > 0:
                # Turnover = Volume / Average Inventory
                turnover = total_volume / gross_exposure
                return turnover
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating inventory turnover: {e}")
            return 0
    
    def get_microstructure_summary(self, symbol: str) -> Dict:
        """Get summary statistics for market microstructure"""
        try:
            if symbol not in self.microstructure:
                return {}
            
            ms = self.microstructure[symbol]
            summary = {}
            
            # Calculate averages
            if ms['avg_spread']:
                summary['avg_spread'] = np.mean(ms['avg_spread'])
                summary['spread_volatility'] = np.std(ms['avg_spread'])
            
            if ms['avg_volume']:
                summary['avg_volume'] = np.mean(ms['avg_volume'])
                
            if ms['volatility']:
                summary['avg_volatility'] = np.mean(ms['volatility'])
                
            if ms['order_imbalance']:
                summary['avg_imbalance'] = np.mean(ms['order_imbalance'])
                summary['imbalance_std'] = np.std(ms['order_imbalance'])
            
            summary['last_update'] = ms['last_update'].isoformat() if ms['last_update'] else None
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting microstructure summary: {e}")
            return {}
    
    def save_hourly_snapshot(self, inventory_pnl: float):
        """Save hourly performance snapshot"""
        try:
            snapshot = {
                'timestamp': pd.Timestamp.now(),
                'trades': self.performance['total_trades'],
                'volume': self.performance['buy_volume'] + self.performance['sell_volume'],
                'gross_pnl': self.performance['spread_captured'] + self.performance['fee_revenue'],
                'inventory_pnl': inventory_pnl,
                'spread_captured': self.performance['spread_captured'],
                'fee_revenue': self.performance['fee_revenue'],
                'quote_failures': self.performance['quote_failures']
            }
            
            self.hourly_metrics.append(snapshot)
            
            # Also save daily snapshot if new day
            if not self.daily_metrics or snapshot['timestamp'].date() != self.daily_metrics[-1]['timestamp'].date():
                self.daily_metrics.append(snapshot.copy())
                
        except Exception as e:
            logger.error(f"Error saving hourly snapshot: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            gross_pnl = self.performance['spread_captured'] + self.performance['fee_revenue']
            total_volume = self.performance['buy_volume'] + self.performance['sell_volume']
            
            # Calculate additional metrics
            avg_spread_captured = (
                self.performance['spread_captured'] / self.performance['total_trades']
                if self.performance['total_trades'] > 0 else 0
            )
            
            avg_fill_rate = (
                np.mean(self.trade_stats['fill_rates']) 
                if self.trade_stats['fill_rates'] else 0
            )
            
            # Calculate recent performance
            recent_pnl = 0
            recent_volume = 0
            if len(self.hourly_metrics) >= 2:
                recent_pnl = self.hourly_metrics[-1]['gross_pnl'] - self.hourly_metrics[-2]['gross_pnl']
                recent_volume = self.hourly_metrics[-1]['volume'] - self.hourly_metrics[-2]['volume']
            
            return {
                **self.performance,
                'gross_pnl': gross_pnl,
                'net_pnl': gross_pnl + self.performance['inventory_pnl'],
                'total_volume': total_volume,
                'avg_spread_captured': avg_spread_captured,
                'avg_fill_rate': avg_fill_rate,
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'recent_hourly_pnl': recent_pnl,
                'recent_hourly_volume': recent_volume,
                'success_rate': 1 - (self.performance['quote_failures'] / 
                                   max(1, self.performance['total_trades'] + self.performance['quote_failures']))
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return self.performance.copy()
    
    def get_time_series_metrics(self, period: str = 'hourly') -> List[Dict]:
        """Get time series of performance metrics"""
        try:
            if period == 'hourly':
                metrics = list(self.hourly_metrics)
            elif period == 'daily':
                metrics = list(self.daily_metrics)
            else:
                raise ValueError(f"Unknown period: {period}")
            
            # Convert timestamps to strings for serialization
            for metric in metrics:
                if 'timestamp' in metric and isinstance(metric['timestamp'], pd.Timestamp):
                    metric['timestamp'] = metric['timestamp'].isoformat()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting time series metrics: {e}")
            return []
    
    def reset_metrics(self):
        """Reset performance metrics (use with caution)"""
        logger.warning("Resetting performance metrics")
        
        self.performance = {
            'total_trades': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'gross_pnl': 0,
            'fee_revenue': 0,
            'inventory_pnl': 0,
            'spread_captured': 0,
            'quote_failures': 0,
            'errors': 0
        }
        
        self.trade_stats = {
            'spreads_captured': deque(maxlen=1000),
            'fill_rates': deque(maxlen=100),
            'quote_lifetimes': deque(maxlen=1000)
        }