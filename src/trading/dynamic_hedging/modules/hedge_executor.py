"""
Hedge Executor Module

Handles the execution of hedge recommendations.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from .hedge_types import HedgeRecommendation, HedgePosition, HedgeStatus

logger = logging.getLogger(__name__)


class HedgeExecutor:
    """
    Executes hedge recommendations through the order execution system.
    
    Manages the placement and monitoring of hedge orders, ensuring
    proper execution and tracking of hedge positions.
    """
    
    def __init__(self):
        """Initialize the hedge executor."""
        self.execution_history = []
        self.active_executions = {}
    
    async def execute_hedge(
        self,
        recommendation: HedgeRecommendation,
        order_executor: Any
    ) -> Dict[str, Any]:
        """
        Execute a hedge recommendation.
        
        Args:
            recommendation: Hedge recommendation to execute
            order_executor: Order execution interface
            
        Returns:
            Execution result dictionary
        """
        try:
            # Validate recommendation
            validation_result = self._validate_recommendation(recommendation)
            if not validation_result['valid']:
                return {
                    'status': 'error',
                    'error': validation_result['reason']
                }
            
            # Prepare hedge order
            order_params = self._prepare_order_params(recommendation)
            
            logger.info(
                f"Executing {recommendation.hedge_type} hedge: "
                f"{recommendation.symbol} {recommendation.side} "
                f"{recommendation.size:.4f}"
            )
            
            # Execute order
            order_result = await order_executor.place_order(**order_params)
            
            if order_result.get('status') in ['filled', 'partial']:
                # Create hedge position
                hedge_position = self._create_hedge_position(
                    recommendation,
                    order_result
                )
                
                # Record execution
                self._record_execution(hedge_position, order_result)
                
                logger.info(
                    f"Hedge executed successfully: {hedge_position['hedge_id']}"
                )
                
                return {
                    'status': 'success',
                    'hedge_id': hedge_position['hedge_id'],
                    'hedge_position': hedge_position,
                    'order_result': order_result
                }
            else:
                logger.error(
                    f"Hedge order failed: {order_result.get('reason', 'unknown')}"
                )
                return {
                    'status': 'failed',
                    'reason': 'Order not filled',
                    'order_result': order_result
                }
                
        except Exception as e:
            logger.error(f"Error executing hedge: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _validate_recommendation(
        self,
        recommendation: HedgeRecommendation
    ) -> Dict[str, Any]:
        """Validate hedge recommendation."""
        try:
            # Size validation
            if recommendation.size <= 0 or not np.isfinite(recommendation.size):
                return {
                    'valid': False,
                    'reason': f'Invalid hedge size: {recommendation.size}'
                }
            
            # Side validation
            if recommendation.side not in ['buy', 'sell']:
                return {
                    'valid': False,
                    'reason': f'Invalid side: {recommendation.side}'
                }
            
            # Symbol validation
            if not recommendation.symbol or len(recommendation.symbol) < 3:
                return {
                    'valid': False,
                    'reason': f'Invalid symbol: {recommendation.symbol}'
                }
            
            # Cost validation
            if recommendation.expected_cost < 0:
                return {
                    'valid': False,
                    'reason': 'Negative expected cost'
                }
            
            # Hedge ratio validation
            if not 0 < recommendation.hedge_ratio <= 1:
                return {
                    'valid': False,
                    'reason': f'Invalid hedge ratio: {recommendation.hedge_ratio}'
                }
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Error validating recommendation: {e}")
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}'
            }
    
    def _prepare_order_params(
        self,
        recommendation: HedgeRecommendation
    ) -> Dict[str, Any]:
        """Prepare order parameters from recommendation."""
        # Base order parameters
        params = {
            'symbol': recommendation.symbol,
            'side': recommendation.side,
            'size': recommendation.size,
            'order_type': 'market',  # Use market orders for hedges
            'metadata': {
                'hedge_type': recommendation.hedge_type,
                'hedge_ratio': recommendation.hedge_ratio,
                'reason': recommendation.reason,
                'expected_protection': recommendation.expected_protection
            }
        }
        
        # Add specific parameters based on hedge type
        if recommendation.hedge_type == 'tail_risk_hedge':
            # Use limit orders for options
            params['order_type'] = 'limit'
            params['price'] = recommendation.metadata.get('strike_price', 0)
            params['time_in_force'] = 'GTC'
        
        elif recommendation.hedge_type in ['beta_hedge', 'volatility_hedge']:
            # Use IOC for immediate execution
            params['time_in_force'] = 'IOC'
        
        return params
    
    def _create_hedge_position(
        self,
        recommendation: HedgeRecommendation,
        order_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create hedge position from execution result."""
        hedge_id = f"hedge_{recommendation.hedge_type}_{int(datetime.now().timestamp())}"
        
        position = {
            'hedge_id': hedge_id,
            'hedge_type': recommendation.hedge_type,
            'symbol': recommendation.symbol,
            'side': recommendation.side,
            'size': order_result.get('filled_size', recommendation.size),
            'entry_price': order_result.get('fill_price', 0),
            'hedge_ratio': recommendation.hedge_ratio,
            'timestamp': datetime.now(),
            'status': 'active',
            'reason': recommendation.reason,
            'expected_cost': recommendation.expected_cost,
            'expected_protection': recommendation.expected_protection,
            'actual_cost': order_result.get('filled_size', 0) * order_result.get('fill_price', 0) * 0.001,  # Estimate
            'order_id': order_result.get('order_id'),
            'urgency': recommendation.urgency,
            'target_metric': recommendation.target_metric,
            'threshold_breached': recommendation.threshold_breached
        }
        
        return position
    
    def _record_execution(
        self,
        hedge_position: Dict[str, Any],
        order_result: Dict[str, Any]
    ) -> None:
        """Record hedge execution for analysis."""
        execution_record = {
            'timestamp': datetime.now(),
            'hedge_id': hedge_position['hedge_id'],
            'hedge_type': hedge_position['hedge_type'],
            'symbol': hedge_position['symbol'],
            'side': hedge_position['side'],
            'size': hedge_position['size'],
            'price': hedge_position['entry_price'],
            'expected_cost': hedge_position['expected_cost'],
            'execution_time': order_result.get('execution_time', 0),
            'slippage': order_result.get('slippage', 0),
            'fees': order_result.get('fees', 0)
        }
        
        self.execution_history.append(execution_record)
        
        # Limit history size
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]
    
    async def execute_multiple_hedges(
        self,
        recommendations: List[HedgeRecommendation],
        order_executor: Any,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple hedge recommendations concurrently.
        
        Args:
            recommendations: List of hedge recommendations
            order_executor: Order execution interface
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of execution results
        """
        results = []
        
        # Execute in batches to avoid overwhelming the system
        for i in range(0, len(recommendations), max_concurrent):
            batch = recommendations[i:i + max_concurrent]
            
            # Execute batch concurrently
            tasks = [
                self.execute_hedge(rec, order_executor)
                for rec in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Hedge execution failed: {result}")
                    results.append({
                        'status': 'error',
                        'error': str(result),
                        'recommendation': batch[j].to_dict()
                    })
                else:
                    results.append(result)
        
        return results
    
    async def close_hedge(
        self,
        hedge_id: str,
        hedge_position: Dict[str, Any],
        order_executor: Any,
        reason: str = 'manual'
    ) -> Dict[str, Any]:
        """
        Close a hedge position.
        
        Args:
            hedge_id: Hedge position ID
            hedge_position: Hedge position details
            order_executor: Order execution interface
            reason: Reason for closing
            
        Returns:
            Closing result
        """
        try:
            # Prepare closing order (opposite side)
            closing_side = 'sell' if hedge_position['side'] == 'buy' else 'buy'
            
            order_params = {
                'symbol': hedge_position['symbol'],
                'side': closing_side,
                'size': hedge_position['size'],
                'order_type': 'market',
                'metadata': {
                    'hedge_id': hedge_id,
                    'closing_reason': reason
                }
            }
            
            logger.info(f"Closing hedge {hedge_id}: {reason}")
            
            # Execute closing order
            order_result = await order_executor.place_order(**order_params)
            
            if order_result.get('status') in ['filled', 'partial']:
                # Calculate final P&L
                exit_price = order_result.get('fill_price', 0)
                
                if hedge_position['side'] == 'buy':
                    pnl = (exit_price - hedge_position['entry_price']) * hedge_position['size']
                else:
                    pnl = (hedge_position['entry_price'] - exit_price) * hedge_position['size']
                
                return {
                    'status': 'success',
                    'hedge_id': hedge_id,
                    'exit_price': exit_price,
                    'realized_pnl': pnl,
                    'closing_reason': reason,
                    'order_result': order_result
                }
            else:
                return {
                    'status': 'failed',
                    'reason': 'Closing order failed',
                    'order_result': order_result
                }
                
        except Exception as e:
            logger.error(f"Error closing hedge: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get hedge execution analytics."""
        if not self.execution_history:
            return {'message': 'No execution history'}
        
        try:
            # Group by hedge type
            by_type = {}
            
            for execution in self.execution_history:
                hedge_type = execution['hedge_type']
                
                if hedge_type not in by_type:
                    by_type[hedge_type] = {
                        'count': 0,
                        'total_size': 0,
                        'total_cost': 0,
                        'avg_execution_time': 0,
                        'avg_slippage': 0
                    }
                
                stats = by_type[hedge_type]
                stats['count'] += 1
                stats['total_size'] += execution['size']
                stats['total_cost'] += execution['expected_cost']
                stats['avg_execution_time'] += execution['execution_time']
                stats['avg_slippage'] += abs(execution['slippage'])
            
            # Calculate averages
            for stats in by_type.values():
                if stats['count'] > 0:
                    stats['avg_execution_time'] /= stats['count']
                    stats['avg_slippage'] /= stats['count']
            
            return {
                'total_executions': len(self.execution_history),
                'by_hedge_type': by_type,
                'recent_executions': self.execution_history[-10:]
            }
            
        except Exception as e:
            logger.error(f"Error calculating execution analytics: {e}")
            return {'error': str(e)}
