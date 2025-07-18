"""
Execution Analytics Module

Tracks and analyzes order execution performance metrics.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta

from .order_types import Order, OrderStatus, ExecutionReport

logger = logging.getLogger(__name__)


class ExecutionAnalytics:
    """
    Analyzes order execution performance and provides insights.
    
    Tracks execution quality, performance metrics, and provides
    analytics for improving execution strategies.
    """
    
    def __init__(self):
        """Initialize execution analytics."""
        self.execution_stats = self._initialize_stats()
        self.performance_history = []
        self.error_log = []
        self.venue_statistics = defaultdict(lambda: defaultdict(float))
        
    def _initialize_stats(self) -> Dict[str, Any]:
        """Initialize execution statistics."""
        return {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'cancelled_orders': 0,
            'partial_fills': 0,
            'avg_slippage': 0,
            'avg_fill_time': 0,
            'total_fees': 0,
            'total_volume': 0,
            'error_count': 0,
            'fill_rate': 0,
            'rejection_rate': 0,
            'slippage_by_symbol': defaultdict(list),
            'fill_time_by_symbol': defaultdict(list),
            'volume_by_symbol': defaultdict(float),
            'orders_by_hour': defaultdict(int),
            'performance_by_side': {
                'buy': {'count': 0, 'avg_slippage': 0},
                'sell': {'count': 0, 'avg_slippage': 0}
            }
        }
    
    def update_execution_stats(
        self,
        order: Order,
        execution_time: float
    ) -> None:
        """
        Update execution statistics with completed order.
        
        Args:
            order: Completed order
            execution_time: Time taken to execute
        """
        try:
            self.execution_stats['total_orders'] += 1
            
            # Update status-based counters
            if order.status == OrderStatus.FILLED:
                self.execution_stats['filled_orders'] += 1
            elif order.status == OrderStatus.REJECTED:
                self.execution_stats['rejected_orders'] += 1
            elif order.status == OrderStatus.CANCELLED:
                self.execution_stats['cancelled_orders'] += 1
            elif order.status == OrderStatus.PARTIAL:
                self.execution_stats['partial_fills'] += 1
            
            # Update averages using running average formula
            if order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
                # Slippage
                if 'expected_price' in order.metadata:
                    slippage = self._calculate_slippage(
                        order,
                        order.metadata['expected_price']
                    )
                    self._update_running_average(
                        'avg_slippage',
                        slippage,
                        self.execution_stats['filled_orders']
                    )
                    
                    # Track by symbol
                    self.execution_stats['slippage_by_symbol'][order.symbol].append(slippage)
                    
                    # Track by side
                    side_stats = self.execution_stats['performance_by_side'][order.side]
                    side_stats['count'] += 1
                    if side_stats['count'] == 1:
                        side_stats['avg_slippage'] = slippage
                    else:
                        side_stats['avg_slippage'] = (
                            (side_stats['avg_slippage'] * (side_stats['count'] - 1) + slippage) /
                            side_stats['count']
                        )
                
                # Fill time
                self._update_running_average(
                    'avg_fill_time',
                    execution_time,
                    self.execution_stats['filled_orders']
                )
                self.execution_stats['fill_time_by_symbol'][order.symbol].append(execution_time)
                
                # Volume and fees
                volume = order.filled_size * order.avg_fill_price
                self.execution_stats['total_volume'] += volume
                self.execution_stats['volume_by_symbol'][order.symbol] += volume
                
                if hasattr(order, 'fees_paid'):
                    self.execution_stats['total_fees'] += order.fees_paid
            
            # Update rates
            total = self.execution_stats['total_orders']
            if total > 0:
                self.execution_stats['fill_rate'] = (
                    self.execution_stats['filled_orders'] / total
                )
                self.execution_stats['rejection_rate'] = (
                    self.execution_stats['rejected_orders'] / total
                )
            
            # Track by hour
            hour = datetime.fromtimestamp(order.timestamp).hour
            self.execution_stats['orders_by_hour'][hour] += 1
            
            # Add to performance history
            self._add_to_history(order, execution_time)
            
        except Exception as e:
            logger.error(f"Error updating execution stats: {e}")
            self.record_error('stats_update', str(e))
    
    def _update_running_average(
        self,
        stat_name: str,
        new_value: float,
        count: int
    ) -> None:
        """Update running average for a statistic."""
        if count <= 0:
            return
            
        if count == 1:
            self.execution_stats[stat_name] = new_value
        else:
            old_avg = self.execution_stats[stat_name]
            self.execution_stats[stat_name] = (
                (old_avg * (count - 1) + new_value) / count
            )
    
    def _calculate_slippage(
        self,
        order: Order,
        expected_price: float
    ) -> float:
        """Calculate slippage for an order."""
        if expected_price <= 0 or order.avg_fill_price <= 0:
            return 0
        
        if order.side == 'buy':
            return (order.avg_fill_price - expected_price) / expected_price
        else:
            return (expected_price - order.avg_fill_price) / expected_price
    
    def _add_to_history(
        self,
        order: Order,
        execution_time: float
    ) -> None:
        """Add order to performance history."""
        record = {
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': order.side,
            'size': order.size,
            'filled_size': order.filled_size,
            'status': order.status.value,
            'execution_time': execution_time,
            'slippage': order.slippage if hasattr(order, 'slippage') else 0,
            'fill_rate': order.fill_percentage
        }
        
        self.performance_history.append(record)
        
        # Limit history size
        if len(self.performance_history) > 10000:
            self.performance_history = self.performance_history[-5000:]
    
    def record_rejection(self, order: Order) -> None:
        """Record order rejection for analysis."""
        try:
            rejection_record = {
                'timestamp': time.time(),
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side,
                'size': order.size,
                'reason': order.rejection_reason or 'unknown'
            }
            
            # Could implement rejection pattern analysis here
            logger.info(f"Order rejected: {rejection_record}")
            
        except Exception as e:
            logger.error(f"Error recording rejection: {e}")
    
    def record_error(self, error_type: str, error_message: str) -> None:
        """Record execution error."""
        self.execution_stats['error_count'] += 1
        
        error_record = {
            'timestamp': time.time(),
            'type': error_type,
            'message': error_message
        }
        
        self.error_log.append(error_record)
        
        # Limit error log size
        if len(self.error_log) > 1000:
            self.error_log = self.error_log[-500:]
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive execution analytics.
        
        Returns:
            Analytics dictionary
        """
        try:
            total_orders = self.execution_stats.get('total_orders', 0)
            
            if total_orders == 0:
                return {
                    'message': 'No orders executed yet',
                    'total_orders': 0
                }
            
            # Calculate additional metrics
            analytics = {
                'summary': {
                    'total_orders': total_orders,
                    'fill_rate': self.execution_stats['fill_rate'],
                    'avg_slippage_bps': self.execution_stats['avg_slippage'] * 10000,
                    'avg_fill_time_ms': self.execution_stats['avg_fill_time'] * 1000,
                    'total_volume': self.execution_stats['total_volume'],
                    'total_fees': self.execution_stats['total_fees'],
                    'fee_rate_bps': (
                        self.execution_stats['total_fees'] /
                        self.execution_stats['total_volume'] * 10000
                        if self.execution_stats['total_volume'] > 0 else 0
                    )
                },
                'performance': {
                    'filled_orders': self.execution_stats['filled_orders'],
                    'rejected_orders': self.execution_stats['rejected_orders'],
                    'cancelled_orders': self.execution_stats['cancelled_orders'],
                    'partial_fills': self.execution_stats['partial_fills'],
                    'rejection_rate': self.execution_stats['rejection_rate'],
                    'error_rate': self.execution_stats['error_count'] / total_orders
                },
                'by_symbol': self._get_symbol_analytics(),
                'by_side': self.execution_stats['performance_by_side'],
                'by_hour': dict(self.execution_stats['orders_by_hour']),
                'recent_errors': self.error_log[-10:] if self.error_log else []
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting execution analytics: {e}")
            return {'error': str(e)}
    
    def _get_symbol_analytics(self) -> Dict[str, Dict[str, Any]]:
        """Get per-symbol analytics."""
        symbol_analytics = {}
        
        for symbol in self.execution_stats['volume_by_symbol'].keys():
            slippages = self.execution_stats['slippage_by_symbol'].get(symbol, [])
            fill_times = self.execution_stats['fill_time_by_symbol'].get(symbol, [])
            
            if slippages:
                symbol_analytics[symbol] = {
                    'volume': self.execution_stats['volume_by_symbol'][symbol],
                    'order_count': len(slippages),
                    'avg_slippage_bps': np.mean(slippages) * 10000,
                    'max_slippage_bps': max(slippages) * 10000,
                    'avg_fill_time_ms': np.mean(fill_times) * 1000 if fill_times else 0
                }
        
        return symbol_analytics
    
    def get_performance_report(
        self,
        time_window: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate performance report.
        
        Args:
            time_window: Time window in seconds
            symbol: Optional symbol filter
            
        Returns:
            Performance report
        """
        try:
            # Filter history
            records = self.performance_history
            
            if time_window:
                cutoff = time.time() - time_window
                records = [r for r in records if r['timestamp'] > cutoff]
            
            if symbol:
                records = [r for r in records if r['symbol'] == symbol]
            
            if not records:
                return {'message': 'No data for specified criteria'}
            
            # Calculate metrics
            fill_rates = [r['fill_rate'] for r in records]
            execution_times = [r['execution_time'] for r in records]
            slippages = [r['slippage'] for r in records if r['slippage'] != 0]
            
            return {
                'period': {
                    'start': datetime.fromtimestamp(records[0]['timestamp']).isoformat(),
                    'end': datetime.fromtimestamp(records[-1]['timestamp']).isoformat(),
                    'order_count': len(records)
                },
                'fill_quality': {
                    'avg_fill_rate': np.mean(fill_rates),
                    'min_fill_rate': min(fill_rates),
                    'full_fills': sum(1 for r in fill_rates if r == 100),
                    'partial_fills': sum(1 for r in fill_rates if 0 < r < 100)
                },
                'execution_speed': {
                    'avg_time_ms': np.mean(execution_times) * 1000,
                    'median_time_ms': np.median(execution_times) * 1000,
                    'p95_time_ms': np.percentile(execution_times, 95) * 1000
                },
                'slippage_analysis': {
                    'avg_slippage_bps': np.mean(slippages) * 10000 if slippages else 0,
                    'positive_slippage_rate': (
                        sum(1 for s in slippages if s > 0) / len(slippages)
                        if slippages else 0
                    ),
                    'slippage_std_bps': np.std(slippages) * 10000 if slippages else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get suggestions for execution optimization.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        try:
            # High slippage suggestion
            if self.execution_stats['avg_slippage'] > 0.002:  # 20 bps
                suggestions.append({
                    'issue': 'High average slippage',
                    'current': f"{self.execution_stats['avg_slippage'] * 10000:.1f} bps",
                    'suggestion': 'Consider using TWAP or Iceberg orders for large trades',
                    'priority': 'high'
                })
            
            # Low fill rate suggestion
            if self.execution_stats['fill_rate'] < 0.9:
                suggestions.append({
                    'issue': 'Low fill rate',
                    'current': f"{self.execution_stats['fill_rate']:.1%}",
                    'suggestion': 'Review limit order pricing or use more aggressive strategies',
                    'priority': 'medium'
                })
            
            # Slow execution suggestion
            if self.execution_stats['avg_fill_time'] > 10:  # 10 seconds
                suggestions.append({
                    'issue': 'Slow average fill time',
                    'current': f"{self.execution_stats['avg_fill_time']:.1f} seconds",
                    'suggestion': 'Consider more aggressive pricing or IOC orders',
                    'priority': 'medium'
                })
            
            # High rejection rate
            if self.execution_stats['rejection_rate'] > 0.05:
                suggestions.append({
                    'issue': 'High rejection rate',
                    'current': f"{self.execution_stats['rejection_rate']:.1%}",
                    'suggestion': 'Review order validation and risk limits',
                    'priority': 'high'
                })
            
            # Symbol concentration
            symbol_volumes = self.execution_stats['volume_by_symbol']
            if symbol_volumes:
                total_volume = sum(symbol_volumes.values())
                max_symbol_volume = max(symbol_volumes.values())
                concentration = max_symbol_volume / total_volume
                
                if concentration > 0.8:
                    suggestions.append({
                        'issue': 'High symbol concentration',
                        'current': f"{concentration:.1%} in one symbol",
                        'suggestion': 'Consider diversifying trading across more symbols',
                        'priority': 'low'
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating optimization suggestions: {e}")
            return []
