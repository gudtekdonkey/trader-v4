"""
Portfolio report generation
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from .data_types import PortfolioMetrics, RebalancingRecommendation

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates portfolio analysis reports"""
    
    def generate_report(self,
                       positions: Dict,
                       metrics: Optional[PortfolioMetrics],
                       risk_metrics: Optional[Dict],
                       rebalancing_recommendations: Optional[List[RebalancingRecommendation]],
                       regime_info: Optional[Dict]) -> Dict:
        """Generate comprehensive portfolio analysis report"""
        try:
            report = {
                'timestamp': datetime.now(),
                'portfolio_overview': self._generate_overview(positions),
                'performance_metrics': {},
                'risk_analysis': {},
                'rebalancing_analysis': {},
                'regime_analysis': {},
                'alerts': []
            }
            
            # Add performance metrics if available
            if metrics:
                report['performance_metrics'] = {
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'volatility': metrics.volatility,
                    'win_rate': metrics.win_rate,
                    'var_95': metrics.var_95
                }
            
            # Add risk analysis if available
            if risk_metrics:
                report['risk_analysis'] = risk_metrics
            
            # Add rebalancing analysis if available
            if rebalancing_recommendations:
                report['rebalancing_analysis'] = self._generate_rebalancing_summary(
                    rebalancing_recommendations
                )
            
            # Add regime analysis if available
            if regime_info:
                report['regime_analysis'] = regime_info
            
            # Generate alerts
            report['alerts'] = self._generate_alerts(
                positions, 
                report['performance_metrics'],
                report['rebalancing_analysis'],
                regime_info
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating portfolio report: {e}")
            return {'error': str(e)}
    
    def _generate_overview(self, positions: Dict) -> Dict:
        """Generate portfolio overview"""
        total_value = sum(
            pos['size'] * pos.get('current_price', pos['entry_price']) 
            for pos in positions.values()
        )
        
        current_weights = {}
        for symbol, position in positions.items():
            current_price = position.get('current_price', position['entry_price'])
            value = position['size'] * current_price
            current_weights[symbol] = value / total_value if total_value > 0 else 0
        
        overview = {
            'total_value': total_value,
            'number_of_positions': len(positions),
            'current_weights': current_weights
        }
        
        if current_weights:
            overview['largest_position'] = max(current_weights.items(), key=lambda x: x[1])
            overview['smallest_position'] = min(current_weights.items(), key=lambda x: x[1])
        
        return overview
    
    def _generate_rebalancing_summary(self, 
                                    recommendations: List[RebalancingRecommendation]) -> Dict:
        """Generate rebalancing summary"""
        high_priority = sum(1 for r in recommendations if r.urgency == 'high')
        
        return {
            'recommendations_count': len(recommendations),
            'high_priority_count': high_priority,
            'recommendations': [
                {
                    'symbol': r.symbol,
                    'action': r.action,
                    'current_weight': r.current_weight,
                    'target_weight': r.target_weight,
                    'urgency': r.urgency,
                    'reason': r.reason
                } for r in recommendations[:5]  # Top 5 recommendations
            ]
        }
    
    def _generate_alerts(self, positions: Dict, performance: Dict, 
                        rebalancing: Dict, regime_info: Optional[Dict]) -> List[Dict]:
        """Generate alerts based on portfolio analysis"""
        alerts = []
        
        # Concentration risk alert
        if positions:
            weights = [
                pos['size'] * pos.get('current_price', pos['entry_price']) 
                for pos in positions.values()
            ]
            total_value = sum(weights)
            if total_value > 0:
                max_weight = max(weights) / total_value
                if max_weight > 0.4:
                    alerts.append({
                        'type': 'concentration_risk',
                        'severity': 'high' if max_weight > 0.6 else 'medium',
                        'message': f"High concentration risk: {max_weight:.1%} in single position"
                    })
        
        # Performance alerts
        if performance:
            if performance.get('max_drawdown', 0) > 0.15:
                alerts.append({
                    'type': 'drawdown',
                    'severity': 'high',
                    'message': f"High drawdown: {performance['max_drawdown']:.1%}"
                })
            
            if performance.get('sharpe_ratio', 0) < 0:
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Negative risk-adjusted returns: Sharpe {performance['sharpe_ratio']:.2f}"
                })
        
        # Rebalancing alerts
        if rebalancing and rebalancing.get('high_priority_count', 0) > 0:
            alerts.append({
                'type': 'rebalancing',
                'severity': 'medium',
                'message': f"{rebalancing['high_priority_count']} high-priority rebalancing actions needed"
            })
        
        # Regime alerts
        if regime_info and regime_info.get('stress_level') == 'high':
            alerts.append({
                'type': 'market_stress',
                'severity': 'high',
                'message': "High market stress detected"
            })
        
        return alerts
