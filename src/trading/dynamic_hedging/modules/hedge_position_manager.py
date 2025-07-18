"""
Hedge Position Manager Module

Manages hedge positions including tracking, updates, and lifecycle.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from .hedge_types import HedgePosition, HedgeStatus, HedgeRecommendation

logger = logging.getLogger(__name__)


class HedgePositionManager:
    """
    Manages the lifecycle of hedge positions.
    
    Tracks active hedges, maintains position history, and provides
    position analytics and reporting capabilities.
    """
    
    def __init__(self):
        """Initialize the hedge position manager."""
        self.active_positions: Dict[str, HedgePosition] = {}
        self.position_history: List[HedgePosition] = []
        self.position_by_type: Dict[str, List[str]] = defaultdict(list)
        self.performance_metrics = defaultdict(lambda: {
            'total_protection': 0,
            'total_cost': 0,
            'total_pnl': 0,
            'count': 0
        })
    
    def register_hedge_position(
        self,
        hedge_id: str,
        recommendation: HedgeRecommendation,
        execution_details: Dict[str, Any]
    ) -> HedgePosition:
        """
        Register a new hedge position.
        
        Args:
            hedge_id: Unique hedge identifier
            recommendation: Original hedge recommendation
            execution_details: Execution details from order
            
        Returns:
            Created HedgePosition
        """
        try:
            position = HedgePosition(
                hedge_id=hedge_id,
                hedge_type=recommendation.hedge_type,
                symbol=recommendation.symbol,
                side=recommendation.side,
                size=execution_details.get('size', recommendation.size),
                entry_price=execution_details.get('entry_price', 0),
                current_price=execution_details.get('entry_price', 0),
                unrealized_pnl=0,
                status=HedgeStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                hedge_ratio=recommendation.hedge_ratio,
                expected_cost=recommendation.expected_cost,
                expected_protection=recommendation.expected_protection,
                actual_cost=execution_details.get('actual_cost', recommendation.expected_cost),
                actual_protection=0,
                reason=recommendation.reason,
                metadata={
                    'urgency': recommendation.urgency,
                    'target_metric': recommendation.target_metric,
                    'threshold_breached': recommendation.threshold_breached,
                    'order_id': execution_details.get('order_id')
                }
            )
            
            # Register position
            self.active_positions[hedge_id] = position
            self.position_by_type[recommendation.hedge_type].append(hedge_id)
            
            logger.info(f"Registered hedge position: {hedge_id}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error registering hedge position: {e}")
            raise
    
    def update_positions(self, current_prices: Dict[str, float]) -> None:
        """
        Update all active positions with current prices.
        
        Args:
            current_prices: Current market prices
        """
        for hedge_id, position in self.active_positions.items():
            try:
                symbol = position.symbol
                
                # Get current price
                current_price = current_prices.get(symbol)
                
                if current_price is None or not isinstance(current_price, (int, float)) or current_price <= 0:
                    logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue
                
                # Update position
                position.calculate_pnl(current_price)
                position.updated_at = datetime.now()
                
                # Update actual protection based on P&L
                if position.unrealized_pnl > 0:
                    position.actual_protection = position.unrealized_pnl
                
                # Check if position should be closed
                if self._should_auto_close(position):
                    logger.info(f"Auto-closing hedge {hedge_id}")
                    self.close_position(hedge_id, 'auto_close')
                
            except Exception as e:
                logger.error(f"Error updating position {hedge_id}: {e}")
    
    def _should_auto_close(self, position: HedgePosition) -> bool:
        """Determine if position should be auto-closed."""
        # Close if protection target achieved
        if position.actual_protection >= position.expected_protection * 0.9:
            return True
        
        # Close if losing more than 50% of expected cost
        if position.unrealized_pnl < -position.expected_cost * 1.5:
            return True
        
        # Close if position is very old (30 days)
        age_days = (datetime.now() - position.created_at).days
        if age_days > 30:
            return True
        
        return False
    
    def close_position(
        self,
        hedge_id: str,
        reason: str = 'manual'
    ) -> Dict[str, Any]:
        """
        Close a hedge position.
        
        Args:
            hedge_id: Hedge position ID
            reason: Reason for closing
            
        Returns:
            Closing result
        """
        try:
            if hedge_id not in self.active_positions:
                return {
                    'status': 'error',
                    'error': f'Position {hedge_id} not found'
                }
            
            position = self.active_positions[hedge_id]
            
            # Update status
            position.status = HedgeStatus.CLOSED
            position.updated_at = datetime.now()
            
            # Calculate final metrics
            final_metrics = {
                'hedge_id': hedge_id,
                'realized_pnl': position.unrealized_pnl,
                'actual_protection': position.actual_protection,
                'actual_cost': position.actual_cost,
                'effectiveness': position.effectiveness_ratio(),
                'cost_efficiency': position.cost_efficiency(),
                'duration_hours': (
                    position.updated_at - position.created_at
                ).total_seconds() / 3600,
                'closing_reason': reason
            }
            
            # Update performance metrics
            self._update_performance_metrics(position)
            
            # Move to history
            self.position_history.append(position)
            del self.active_positions[hedge_id]
            
            # Remove from type index
            if hedge_id in self.position_by_type[position.hedge_type]:
                self.position_by_type[position.hedge_type].remove(hedge_id)
            
            logger.info(f"Closed hedge position {hedge_id}: {reason}")
            
            return {
                'status': 'success',
                'metrics': final_metrics
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _update_performance_metrics(self, position: HedgePosition) -> None:
        """Update aggregate performance metrics."""
        metrics = self.performance_metrics[position.hedge_type]
        
        metrics['total_protection'] += position.actual_protection
        metrics['total_cost'] += position.actual_cost
        metrics['total_pnl'] += position.unrealized_pnl
        metrics['count'] += 1
    
    def get_active_positions(self) -> Dict[str, HedgePosition]:
        """Get all active hedge positions."""
        return self.active_positions.copy()
    
    def get_historical_positions(self) -> List[HedgePosition]:
        """Get historical hedge positions."""
        return self.position_history.copy()
    
    def get_positions_by_type(self, hedge_type: str) -> List[HedgePosition]:
        """Get positions by hedge type."""
        position_ids = self.position_by_type.get(hedge_type, [])
        positions = []
        
        # Active positions
        for pos_id in position_ids:
            if pos_id in self.active_positions:
                positions.append(self.active_positions[pos_id])
        
        # Historical positions
        for pos in self.position_history:
            if pos.hedge_type == hedge_type:
                positions.append(pos)
        
        return positions
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive position analytics."""
        try:
            # Active positions summary
            active_summary = {
                'count': len(self.active_positions),
                'total_value': sum(
                    p.size * p.current_price 
                    for p in self.active_positions.values()
                ),
                'total_pnl': sum(
                    p.unrealized_pnl 
                    for p in self.active_positions.values()
                ),
                'by_type': {}
            }
            
            # Group active positions by type
            for hedge_type, position_ids in self.position_by_type.items():
                active_positions = [
                    self.active_positions[pid] 
                    for pid in position_ids 
                    if pid in self.active_positions
                ]
                
                if active_positions:
                    active_summary['by_type'][hedge_type] = {
                        'count': len(active_positions),
                        'total_size': sum(p.size for p in active_positions),
                        'total_pnl': sum(p.unrealized_pnl for p in active_positions),
                        'avg_effectiveness': sum(
                            p.effectiveness_ratio() for p in active_positions
                        ) / len(active_positions)
                    }
            
            # Historical performance
            historical_summary = {
                'total_positions': len(self.position_history),
                'by_type': self.performance_metrics
            }
            
            # Calculate overall metrics
            total_protection = sum(
                m['total_protection'] 
                for m in self.performance_metrics.values()
            )
            total_cost = sum(
                m['total_cost'] 
                for m in self.performance_metrics.values()
            )
            total_pnl = sum(
                m['total_pnl'] 
                for m in self.performance_metrics.values()
            )
            
            overall_metrics = {
                'total_protection_provided': total_protection,
                'total_cost_incurred': total_cost,
                'total_pnl': total_pnl,
                'net_benefit': total_protection - total_cost,
                'roi': (total_pnl / total_cost) if total_cost > 0 else 0,
                'protection_efficiency': (
                    total_protection / total_cost 
                    if total_cost > 0 else 0
                )
            }
            
            return {
                'active_positions': active_summary,
                'historical_performance': historical_summary,
                'overall_metrics': overall_metrics,
                'position_history_count': len(self.position_history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating analytics: {e}")
            return {'error': str(e)}
    
    def get_position_details(self, hedge_id: str) -> Optional[HedgePosition]:
        """Get details of a specific position."""
        # Check active positions
        if hedge_id in self.active_positions:
            return self.active_positions[hedge_id]
        
        # Check history
        for position in self.position_history:
            if position.hedge_id == hedge_id:
                return position
        
        return None
    
    def cleanup_old_history(self, days_to_keep: int = 90) -> int:
        """
        Clean up old position history.
        
        Args:
            days_to_keep: Number of days of history to keep
            
        Returns:
            Number of positions removed
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            old_positions = [
                p for p in self.position_history 
                if p.updated_at < cutoff_date
            ]
            
            self.position_history = [
                p for p in self.position_history 
                if p.updated_at >= cutoff_date
            ]
            
            logger.info(f"Cleaned up {len(old_positions)} old positions")
            return len(old_positions)
            
        except Exception as e:
            logger.error(f"Error cleaning up history: {e}")
            return 0
