"""
Portfolio Analytics Module
Provides advanced portfolio analysis, performance metrics, and rebalancing insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    var_95: float
    cvar_95: float
    diversification_ratio: float
    concentration_risk: float

@dataclass
class RebalancingRecommendation:
    """Rebalancing recommendation for a specific asset"""
    symbol: str
    current_weight: float
    target_weight: float
    weight_deviation: float
    action: str  # 'buy', 'sell', 'hold'
    amount_to_trade: float
    urgency: str  # 'low', 'medium', 'high'
    reason: str

class PortfolioAnalytics:
    """Advanced portfolio analytics and optimization"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.performance_history = []
        
    def calculate_portfolio_metrics(self, 
                                  returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  positions: Dict = None) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Risk-adjusted returns
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Calmar ratio
            calmar_ratio = (total_return / max_drawdown) if max_drawdown > 0 else 0
            
            # Trade-based metrics (if available)
            win_rate = 0
            profit_factor = 0
            if positions:
                closed_trades = [pos for pos in positions.values() if pos.get('status') == 'closed']
                if closed_trades:
                    wins = sum(1 for trade in closed_trades if trade.get('pnl', 0) > 0)
                    win_rate = wins / len(closed_trades)
                    
                    gross_profit = sum(trade.get('pnl', 0) for trade in closed_trades if trade.get('pnl', 0) > 0)
                    gross_loss = abs(sum(trade.get('pnl', 0) for trade in closed_trades if trade.get('pnl', 0) < 0))
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Market-relative metrics
            beta = 0
            alpha = 0
            information_ratio = 0
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                # Beta calculation
                covariance = np.cov(returns, benchmark_returns)[0][1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Alpha calculation (Jensen's alpha)
                expected_return = self.risk_free_rate / 252 + beta * (benchmark_returns.mean() - self.risk_free_rate / 252)
                alpha = returns.mean() - expected_return
                alpha = alpha * 252  # Annualized
                
                # Information ratio
                active_returns = returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_returns.mean() / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
            
            # Value at Risk (VaR) and Conditional VaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            
            # Portfolio concentration and diversification
            diversification_ratio = 0
            concentration_risk = 0
            if positions:
                total_value = sum(pos.get('value', 0) for pos in positions.values())
                if total_value > 0:
                    weights = [pos.get('value', 0) / total_value for pos in positions.values()]
                    # Herfindahl-Hirschman Index for concentration
                    concentration_risk = sum(w**2 for w in weights)
                    # Simple diversification measure
                    diversification_ratio = 1 / concentration_risk if concentration_risk > 0 else 0
            
            return PortfolioMetrics(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                volatility=volatility,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                var_95=var_95,
                cvar_95=cvar_95,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
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
                        urgency = 'low'
                        if abs(weight_deviation) > tolerance * 2:
                            urgency = 'medium'
                        if abs(weight_deviation) > tolerance * 3:
                            urgency = 'high'
                        
                        # Generate reason
                        reason = f"Weight deviation: {weight_deviation:.2%}"
                        if abs(weight_deviation) > 0.1:
                            reason += " (significant drift)"
                        
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
            urgency_order = {'high': 3, 'medium': 2, 'low': 1}
            recommendations.sort(
                key=lambda x: (urgency_order[x.urgency], abs(x.weight_deviation)), 
                reverse=True
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing rebalancing needs: {e}")
            return recommendations
    
    def calculate_portfolio_risk_metrics(self, 
                                       returns_matrix: pd.DataFrame,
                                       weights: np.ndarray) -> Dict:
        """Calculate advanced portfolio risk metrics"""
        try:
            # Portfolio returns
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            
            # Correlation matrix
            correlation_matrix = returns_matrix.corr()
            
            # Portfolio volatility
            portfolio_variance = np.dot(weights.T, np.dot(returns_matrix.cov() * 252, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Component contributions to risk
            marginal_risk = np.dot(returns_matrix.cov() * 252, weights) / portfolio_volatility
            component_risk = weights * marginal_risk
            risk_contribution = component_risk / portfolio_volatility
            
            # Diversification metrics
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            effective_number_assets = 1 / (weights**2).sum()
            
            # Maximum component risk
            max_component_risk = risk_contribution.max()
            
            # Risk concentration
            risk_concentration = (risk_contribution**2).sum()
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'component_risk_contributions': dict(zip(returns_matrix.columns, risk_contribution)),
                'marginal_risk_contributions': dict(zip(returns_matrix.columns, marginal_risk)),
                'average_correlation': avg_correlation,
                'effective_number_assets': effective_number_assets,
                'max_component_risk': max_component_risk,
                'risk_concentration': risk_concentration,
                'correlation_matrix': correlation_matrix
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}
    
    def detect_regime_changes(self, 
                            returns: pd.Series, 
                            window: int = 60) -> Dict:
        """Detect market regime changes that might affect portfolio allocation"""
        try:
            regime_indicators = {}
            
            # Rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            current_vol = rolling_vol.iloc[-1]
            avg_vol = rolling_vol.mean()
            vol_regime = 'high' if current_vol > avg_vol * 1.2 else 'low' if current_vol < avg_vol * 0.8 else 'normal'
            
            # Rolling correlation with trend
            rolling_returns = returns.rolling(window).mean() * 252
            trend_strength = abs(rolling_returns.iloc[-1])
            
            # Market stress indicator (based on drawdowns)
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.rolling(window).max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            current_drawdown = abs(drawdowns.iloc[-1])
            
            stress_level = 'high' if current_drawdown > 0.1 else 'medium' if current_drawdown > 0.05 else 'low'
            
            regime_indicators = {
                'volatility_regime': vol_regime,
                'current_volatility': current_vol,
                'average_volatility': avg_vol,
                'trend_strength': trend_strength,
                'stress_level': stress_level,
                'current_drawdown': current_drawdown,
                'rebalancing_urgency': 'high' if stress_level == 'high' or vol_regime == 'high' else 'normal'
            }
            
            return regime_indicators
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {e}")
            return {}
    
    def generate_portfolio_report(self,
                                positions: Dict,
                                returns: pd.Series,
                                target_weights: Dict = None) -> Dict:
        """Generate comprehensive portfolio analysis report"""
        try:
            report = {
                'timestamp': datetime.now(),
                'portfolio_overview': {},
                'performance_metrics': {},
                'risk_analysis': {},
                'rebalancing_analysis': {},
                'regime_analysis': {},
                'alerts': []
            }
            
            # Portfolio overview
            total_value = sum(
                pos['size'] * pos.get('current_price', pos['entry_price']) 
                for pos in positions.values()
            )
            
            current_weights = {}
            for symbol, position in positions.items():
                current_price = position.get('current_price', position['entry_price'])
                value = position['size'] * current_price
                current_weights[symbol] = value / total_value if total_value > 0 else 0
            
            report['portfolio_overview'] = {
                'total_value': total_value,
                'number_of_positions': len(positions),
                'current_weights': current_weights,
                'largest_position': max(current_weights.items(), key=lambda x: x[1]) if current_weights else None,
                'smallest_position': min(current_weights.items(), key=lambda x: x[1]) if current_weights else None
            }
            
            # Performance metrics
            if len(returns) > 0:
                metrics = self.calculate_portfolio_metrics(returns, positions=positions)
                report['performance_metrics'] = {
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'volatility': metrics.volatility,
                    'win_rate': metrics.win_rate,
                    'var_95': metrics.var_95
                }
            
            # Risk analysis
            if len(positions) > 1:
                symbols = list(current_weights.keys())
                weights = np.array([current_weights[symbol] for symbol in symbols])
                
                # Create dummy returns matrix for risk calculation
                if len(returns) >= 30:  # Need sufficient data
                    returns_matrix = pd.DataFrame({
                        symbol: returns for symbol in symbols  # Simplified - in practice use individual asset returns
                    })
                    risk_metrics = self.calculate_portfolio_risk_metrics(returns_matrix, weights)
                    report['risk_analysis'] = risk_metrics
            
            # Rebalancing analysis
            if target_weights:
                recommendations = self.analyze_rebalancing_needs(positions, target_weights)
                report['rebalancing_analysis'] = {
                    'recommendations_count': len(recommendations),
                    'high_priority_count': sum(1 for r in recommendations if r.urgency == 'high'),
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
            
            # Regime analysis
            if len(returns) >= 60:
                regime_info = self.detect_regime_changes(returns)
                report['regime_analysis'] = regime_info
            
            # Generate alerts
            alerts = []
            
            # Concentration risk alert
            if current_weights:
                max_weight = max(current_weights.values())
                if max_weight > 0.4:
                    alerts.append({
                        'type': 'concentration_risk',
                        'severity': 'high' if max_weight > 0.6 else 'medium',
                        'message': f"High concentration risk: {max_weight:.1%} in single position"
                    })
            
            # Performance alerts
            if 'performance_metrics' in report and report['performance_metrics']:
                if report['performance_metrics']['max_drawdown'] > 0.15:
                    alerts.append({
                        'type': 'drawdown',
                        'severity': 'high',
                        'message': f"High drawdown: {report['performance_metrics']['max_drawdown']:.1%}"
                    })
                
                if report['performance_metrics']['sharpe_ratio'] < 0:
                    alerts.append({
                        'type': 'performance',
                        'severity': 'medium',
                        'message': f"Negative risk-adjusted returns: Sharpe {report['performance_metrics']['sharpe_ratio']:.2f}"
                    })
            
            # Rebalancing alerts
            if 'rebalancing_analysis' in report and report['rebalancing_analysis']['high_priority_count'] > 0:
                alerts.append({
                    'type': 'rebalancing',
                    'severity': 'medium',
                    'message': f"{report['rebalancing_analysis']['high_priority_count']} high-priority rebalancing actions needed"
                })
            
            report['alerts'] = alerts
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating portfolio report: {e}")
            return {'error': str(e)}
    
    def optimize_rebalancing_timing(self,
                                  recommendations: List[RebalancingRecommendation],
                                  market_conditions: Dict) -> List[Dict]:
        """Optimize the timing and execution of rebalancing trades"""
        try:
            optimized_plan = []
            
            # Sort recommendations by urgency and market impact
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

    def track_rebalancing_performance(self,
                                    executed_trades: List[Dict],
                                    original_plan: List[Dict]) -> Dict:
        """Track and analyze rebalancing execution performance"""
        try:
            analysis = {
                'execution_summary': {},
                'cost_analysis': {},
                'timing_analysis': {},
                'deviation_analysis': {}
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
            
            # Store performance for future optimization
            self.performance_history.append({
                'timestamp': datetime.now(),
                'success_rate': success_rate,
                'total_cost': total_cost,
                'execution_time': sum(
                    (trade.get('execution_time', datetime.now()) - trade.get('planned_time', datetime.now())).total_seconds() 
                    for trade in executed_trades
                ) / total_executed if total_executed > 0 else 0
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error tracking rebalancing performance: {e}")
            return {}
