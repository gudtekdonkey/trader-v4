"""
Metrics collection for monitoring
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects portfolio metrics for monitoring"""
    
    async def collect_metrics(self, risk_manager, portfolio_analytics, data_collector) -> Dict:
        """Collect current portfolio metrics for monitoring"""
        try:
            # Get positions from risk manager
            positions = risk_manager.positions
            
            if not positions:
                return {}
            
            # Calculate portfolio value and weights
            total_value = 0
            position_values = {}
            
            for symbol, position in positions.items():
                current_price = position.get('current_price', position['entry_price'])
                value = position['size'] * current_price
                position_values[symbol] = value
                total_value += value
            
            # Calculate weights
            weights = {symbol: value / total_value for symbol, value in position_values.items()}
            
            # Get risk metrics
            risk_metrics = risk_manager.calculate_risk_metrics()
            
            # Get recent returns (simplified - would use actual portfolio returns)
            portfolio_return_today = sum(
                pos.get('unrealized_pnl', 0) for pos in positions.values()
            ) / total_value if total_value > 0 else 0
            
            return {
                'timestamp': datetime.now(),
                'total_value': total_value,
                'position_count': len(positions),
                'weights': weights,
                'risk_metrics': risk_metrics,
                'portfolio_return_today': portfolio_return_today,
                'largest_position_weight': max(weights.values()) if weights else 0,
                'positions': positions
            }
            
        except Exception as e:
            logger.error(f"Error collecting current metrics: {e}")
            return {}
