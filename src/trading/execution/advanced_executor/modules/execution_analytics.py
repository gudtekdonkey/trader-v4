"""
Execution analytics module for performance tracking and statistics.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ExecutionAnalytics:
    """Tracks and analyzes execution performance"""
    
    def __init__(self, executor):
        self.executor = executor
        
    def add_to_history(self, execution_result: Dict):
        """
        Add execution to history with size limit
        
        Args:
            execution_result: Completed execution result
        """
        try:
            self.executor.execution_history.append(execution_result.copy())
            
            # Limit history size
            if len(self.executor.execution_history) > self.executor.max_execution_history:
                # Keep most recent executions
                self.executor.execution_history = self.executor.execution_history[-self.executor.max_execution_history:]
                
        except Exception as e:
            logger.error(f"Error adding to execution history: {e}")
    
    def get_execution_statistics(self) -> Dict:
        """
        Get execution performance statistics
        
        Returns:
            Dictionary of execution statistics
        """
        try:
            if not self.executor.execution_history:
                return {
                    'total_executions': self.executor.execution_metrics['total_executions'],
                    'successful_executions': self.executor.execution_metrics['successful_executions'],
                    'failed_executions': self.executor.execution_metrics['failed_executions'],
                    'partial_executions': self.executor.execution_metrics['partial_executions'],
                    'active_executions': len(self.executor.active_orders)
                }
            
            completed_executions = [e for e in self.executor.execution_history if e.get('status') == 'completed']
            
            if not completed_executions:
                return self.executor.execution_metrics.copy()
            
            # Calculate statistics
            stats = self._calculate_detailed_statistics(completed_executions)
            
            # Add basic metrics
            stats.update({
                'total_executions': self.executor.execution_metrics['total_executions'],
                'successful_executions': self.executor.execution_metrics['successful_executions'],
                'failed_executions': self.executor.execution_metrics['failed_executions'],
                'partial_executions': self.executor.execution_metrics['partial_executions'],
                'active_executions': len(self.executor.active_orders),
                'execution_types': self._get_execution_type_breakdown()
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating execution statistics: {e}")
            return self.executor.execution_metrics.copy()
    
    def _calculate_detailed_statistics(self, completed_executions: List[Dict]) -> Dict:
        """Calculate detailed execution statistics"""
        fill_rates = []
        execution_times = []
        average_slippages = []
        
        for execution in completed_executions:
            # Fill rate
            fill_rate = execution.get('fill_rate', 0)
            if isinstance(fill_rate, (int, float)) and 0 <= fill_rate <= 1:
                fill_rates.append(fill_rate)
            
            # Execution time
            if 'start_time' in execution and 'end_time' in execution:
                try:
                    duration = (execution['end_time'] - execution['start_time']).total_seconds() / 60
                    if duration > 0:
                        execution_times.append(duration)
                except Exception:
                    pass
            
            # Slippage (if limit price available)
            if 'average_price' in execution and 'price' in execution and execution.get('price'):
                try:
                    slippage = abs(execution['average_price'] - execution['price']) / execution['price']
                    if 0 <= slippage <= 1:
                        average_slippages.append(slippage)
                except Exception:
                    pass
        
        return {
            'average_fill_rate': np.mean(fill_rates) if fill_rates else 0,
            'min_fill_rate': np.min(fill_rates) if fill_rates else 0,
            'max_fill_rate': np.max(fill_rates) if fill_rates else 0,
            'average_execution_time_minutes': np.mean(execution_times) if execution_times else 0,
            'min_execution_time_minutes': np.min(execution_times) if execution_times else 0,
            'max_execution_time_minutes': np.max(execution_times) if execution_times else 0,
            'average_slippage': np.mean(average_slippages) if average_slippages else 0,
            'max_slippage': np.max(average_slippages) if average_slippages else 0
        }
    
    def _get_execution_type_breakdown(self) -> Dict[str, int]:
        """Get breakdown of execution types"""
        try:
            breakdown = {'twap': 0, 'vwap': 0, 'iceberg': 0}
            
            for execution in self.executor.execution_history:
                exec_id = execution.get('execution_id', '')
                if exec_id.startswith('twap_'):
                    breakdown['twap'] += 1
                elif exec_id.startswith('vwap_'):
                    breakdown['vwap'] += 1
                elif exec_id.startswith('iceberg_'):
                    breakdown['iceberg'] += 1
                    
            return breakdown
            
        except Exception as e:
            logger.error(f"Error getting execution type breakdown: {e}")
            return {}
    
    def get_execution_report(self, execution_id: str) -> Optional[Dict]:
        """
        Get detailed report for a specific execution
        
        Args:
            execution_id: Execution ID to report on
            
        Returns:
            Detailed execution report or None
        """
        try:
            # Find execution in history or active orders
            execution = None
            
            if execution_id in self.executor.active_orders:
                execution = self.executor.active_orders[execution_id]
            else:
                for exec in self.executor.execution_history:
                    if exec.get('execution_id') == execution_id:
                        execution = exec
                        break
            
            if not execution:
                return None
            
            # Build detailed report
            report = {
                'execution_id': execution_id,
                'symbol': execution.get('symbol'),
                'side': execution.get('side'),
                'total_size': execution.get('total_size'),
                'filled_size': execution.get('total_filled', 0),
                'fill_rate': execution.get('fill_rate', 0),
                'average_price': execution.get('average_price', 0),
                'status': execution.get('status'),
                'start_time': execution.get('start_time'),
                'end_time': execution.get('end_time'),
                'duration_minutes': None,
                'child_orders_count': len(execution.get('child_orders', [])),
                'errors_count': len(execution.get('errors', [])),
                'errors': execution.get('errors', [])
            }
            
            # Calculate duration if possible
            if report['start_time'] and report['end_time']:
                try:
                    duration = (report['end_time'] - report['start_time']).total_seconds() / 60
                    report['duration_minutes'] = round(duration, 2)
                except Exception:
                    pass
            
            # Add child order summary
            child_orders = execution.get('child_orders', [])
            if child_orders:
                filled_orders = [o for o in child_orders if o.get('status') == 'filled']
                partial_orders = [o for o in child_orders if o.get('status') == 'partial']
                failed_orders = [o for o in child_orders if o.get('status') == 'error']
                
                report['child_orders_summary'] = {
                    'total': len(child_orders),
                    'filled': len(filled_orders),
                    'partial': len(partial_orders),
                    'failed': len(failed_orders)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating execution report: {e}")
            return None
    
    def get_symbol_statistics(self, symbol: str) -> Dict:
        """
        Get execution statistics for a specific symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Symbol-specific statistics
        """
        try:
            symbol_executions = [
                e for e in self.executor.execution_history 
                if e.get('symbol') == symbol and e.get('status') == 'completed'
            ]
            
            if not symbol_executions:
                return {
                    'symbol': symbol,
                    'total_executions': 0,
                    'message': 'No completed executions for this symbol'
                }
            
            stats = self._calculate_detailed_statistics(symbol_executions)
            stats['symbol'] = symbol
            stats['total_executions'] = len(symbol_executions)
            
            # Add volume statistics
            total_volume = sum(e.get('total_filled', 0) for e in symbol_executions)
            stats['total_volume_executed'] = total_volume
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating symbol statistics: {e}")
            return {'symbol': symbol, 'error': str(e)}
