"""
Rebalancing analysis and optimization
"""

import logging
from typing import Dict, List
from datetime import datetime, timedelta

from .data_types import RebalancingRecommendation

logger = logging.getLogger(__name__)


class RebalancingAnalyzer:
    """Analyzes and optimizes portfolio rebalancing"""
    
    def analyze_rebalancing_needs(self,
                                current_positions: Dict,
                                target_weights: Dict,
                                tolerance: float = 0.05) -> List[RebalancingRecommendation]:
        """Analyze portfolio and generate rebalancing recommendations"""
        recommendations = []
        
        try:
            # Calculate total portfolio value
            total_value = sum(
                pos['size'] * pos.get('current_price', pos['entry_price']) 
                for pos in current_positions.values()
            )
            
            if total_value == 0:
                return recommendations
            
            # Analyze each position
            for symbol, target_weight in target_weights.items():
                if symbol in current_positions:
                    position = current_positions[symbol]
                    current_price = position.get('current_price', position['entry_price'])
                    current_value = position['size'] * current_price
                    current_weight = current_value / total_value
                    
                    weight_deviation = target_weight - current_weight
                    
                    # Determine action needed
                    if abs(weight_deviation) > tolerance:
                        action = 'buy' if weight_deviation > 0 else 'sell'
                        target_value = total_value * target_weight
                        amount_to_trade = abs(target_value - current_value) / current_price
                        
                        # Determine urgency
                        urgency = self._determine_urgency(weight_deviation, tolerance)
                        
                        # Generate reason
                        reason = self._generate_reason(weight_deviation)
                        
                        recommendations.append(RebalancingRecommendation(
                            symbol=symbol,
                            current_weight=current_weight,
                            target_weight=target_weight,
                            weight_deviation=weight_deviation,
                            action=action,
                            amount_to_trade=amount_to_trade,
                            urgency=urgency,
                            reason=reason
                        ))
                else:
                    # New position needed
                    if target_weight > tolerance:
                        target_value = total_value * target_weight
                        recommendations.append(RebalancingRecommendation(
                            symbol=symbol,
                            current_weight=0.0,
                            target_weight=target_weight,
                            weight_deviation=target_weight,
                            action='buy',
                            amount_to_trade=target_value,  # Will need current price
                            urgency='medium',
                            reason=f"New position required: {target_weight:.2%}"
                        ))
            
            # Check for positions not in target (should be closed)
            for symbol in current_positions:
                if symbol not in target_weights:
                    position = current_positions[symbol]
                    current_price = position.get('current_price', position['entry_price'])
                    current_value = position['size'] * current_price
                    current_weight = current_value / total_value
                    
                    recommendations.append(RebalancingRecommendation(
                        symbol=symbol,
                        current_weight=current_weight,
                        target_weight=0.0,
                        weight_deviation=-current_weight,
                        action='sell',
                        amount_to_trade=position['size'],
                        urgency='low',
                        reason="Position not in target allocation"
                    ))
            
            # Sort by urgency and deviation size
            recommendations = self._sort_recommendations(recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing rebalancing needs: {e}")
            return recommendations
    
    def _determine_urgency(self, weight_deviation: float, tolerance: float) -> str:
        """Determine urgency level based on deviation"""
        abs_deviation = abs(weight_deviation)
        if abs_deviation > tolerance * 3:
            return 'high'
        elif abs_deviation > tolerance * 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_reason(self, weight_deviation: float) -> str:
        """Generate reason for rebalancing"""
        reason = f"Weight deviation: {weight_deviation:.2%}"
        if abs(weight_deviation) > 0.1:
            reason += " (significant drift)"
        return reason
    
    def _sort_recommendations(self, recommendations: List[RebalancingRecommendation]) -> List[RebalancingRecommendation]:
        """Sort recommendations by urgency and deviation"""
        urgency_order = {'high': 3, 'medium': 2, 'low': 1}
        return sorted(
            recommendations,
            key=lambda x: (urgency_order[x.urgency], abs(x.weight_deviation)), 
            reverse=True
        )
    
    def optimize_timing(self,
                       recommendations: List[RebalancingRecommendation],
                       market_conditions: Dict) -> List[Dict]:
        """Optimize the timing and execution of rebalancing trades"""
        try:
            optimized_plan = []
            
            # Sort recommendations by urgency
            high_priority = [r for r in recommendations if r.urgency == 'high']
            medium_priority = [r for r in recommendations if r.urgency == 'medium']
            low_priority = [r for r in recommendations if r.urgency == 'low']
            
            current_time = datetime.now()
            
            # Schedule high priority trades immediately
            for rec in high_priority:
                optimized_plan.append({
                    'recommendation': rec,
                    'scheduled_time': current_time,
                    'execution_method': 'market' if rec.urgency == 'high' else 'limit',
                    'priority': 1
                })
            
            # Schedule medium priority trades with slight delay
            for i, rec in enumerate(medium_priority):
                delay_minutes = i * 15  # 15-minute spacing
                scheduled_time = current_time + timedelta(minutes=delay_minutes)
                
                optimized_plan.append({
                    'recommendation': rec,
                    'scheduled_time': scheduled_time,
                    'execution_method': 'limit',
                    'priority': 2
                })
            
            # Schedule low priority trades for off-peak hours
            for i, rec in enumerate(low_priority):
                # Schedule during next low-activity period
                hours_delay = 2 + (i * 0.5)  # 2+ hours delay with spacing
                scheduled_time = current_time + timedelta(hours=hours_delay)
                
                optimized_plan.append({
                    'recommendation': rec,
                    'scheduled_time': scheduled_time,
                    'execution_method': 'twap',  # Use TWAP for low priority
                    'priority': 3
                })
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"Error optimizing rebalancing timing: {e}")
            return []
    
    def track_performance(self,
                         executed_trades: List[Dict],
                         original_plan: List[Dict]) -> Dict:
        """Track and analyze rebalancing execution performance"""
        try:
            analysis = {
                'execution_summary': {},
                'cost_analysis': {},
                'timing_analysis': {}
            }
            
            # Execution summary
            total_planned = len(original_plan)
            total_executed = len(executed_trades)
            success_rate = total_executed / total_planned if total_planned > 0 else 0
            
            analysis['execution_summary'] = {
                'total_planned_trades': total_planned,
                'total_executed_trades': total_executed,
                'success_rate': success_rate,
                'failed_trades': total_planned - total_executed
            }
            
            # Cost analysis
            total_fees = sum(trade.get('fee', 0) for trade in executed_trades)
            total_slippage = sum(trade.get('slippage', 0) for trade in executed_trades)
            total_cost = total_fees + total_slippage
            
            analysis['cost_analysis'] = {
                'total_fees': total_fees,
                'total_slippage': total_slippage,
                'total_execution_cost': total_cost,
                'avg_fee_per_trade': total_fees / total_executed if total_executed > 0 else 0
            }
            
            # Timing analysis
            if executed_trades:
                execution_delays = []
                for trade in executed_trades:
                    if 'execution_time' in trade and 'planned_time' in trade:
                        delay = (trade['execution_time'] - trade['planned_time']).total_seconds()
                        execution_delays.append(delay)
                
                if execution_delays:
                    analysis['timing_analysis'] = {
                        'avg_execution_delay': sum(execution_delays) / len(execution_delays),
                        'max_delay': max(execution_delays),
                        'min_delay': min(execution_delays)
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error tracking rebalancing performance: {e}")
            return {}
