"""
Dashboard API Routes Module
Handles all API endpoints for portfolio data
"""

from flask import jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class APIRoutes:
    """API route handlers for dashboard"""
    
    def __init__(self, portfolio_analytics, portfolio_monitor, risk_manager):
        self.portfolio_analytics = portfolio_analytics
        self.portfolio_monitor = portfolio_monitor
        self.risk_manager = risk_manager
    
    def portfolio_overview(self):
        """Portfolio overview data endpoint"""
        try:
            positions = self.risk_manager.positions
            
            if not positions:
                return jsonify({'error': 'No positions found'})
            
            # Calculate overview metrics
            total_value = sum(
                pos['size'] * pos.get('current_price', pos['entry_price'])
                for pos in positions.values()
            )
            
            total_pnl = sum(
                pos.get('unrealized_pnl', 0) + pos.get('realized_pnl', 0)
                for pos in positions.values()
            )
            
            positions_data = []
            for symbol, position in positions.items():
                current_price = position.get('current_price', position['entry_price'])
                value = position['size'] * current_price
                weight = value / total_value if total_value > 0 else 0
                
                positions_data.append({
                    'symbol': symbol,
                    'side': position['side'],
                    'size': position['size'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'value': value,
                    'weight': weight,
                    'pnl': position.get('unrealized_pnl', 0),
                    'pnl_pct': (position.get('unrealized_pnl', 0) / (position['size'] * position['entry_price'])) * 100
                })
            
            return jsonify({
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_return_pct': (total_pnl / total_value) * 100 if total_value > 0 else 0,
                'position_count': len(positions),
                'positions': positions_data
            })
            
        except Exception as e:
            logger.error(f"Error getting portfolio overview: {e}")
            return jsonify({'error': str(e)})
    
    def portfolio_performance(self):
        """Portfolio performance metrics endpoint"""
        try:
            # Get portfolio returns (simplified)
            positions = self.risk_manager.positions
            symbols = list(positions.keys())
            
            if not symbols:
                return jsonify({'error': 'No positions for performance calculation'})
            
            # Calculate dummy portfolio returns for demo
            returns = self._generate_demo_returns()
            
            # Calculate metrics
            metrics = self.portfolio_analytics.calculate_portfolio_metrics(
                returns, 
                positions=positions
            )
            
            return jsonify({
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'var_95': metrics.var_95,
                'diversification_ratio': metrics.diversification_ratio
            })
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return jsonify({'error': str(e)})
    
    def portfolio_allocation(self):
        """Portfolio allocation chart data endpoint"""
        try:
            positions = self.risk_manager.positions
            
            if not positions:
                return jsonify({'labels': [], 'values': []})
            
            total_value = sum(
                pos['size'] * pos.get('current_price', pos['entry_price'])
                for pos in positions.values()
            )
            
            labels = []
            values = []
            colors = []
            
            color_palette = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD'
            ]
            
            for i, (symbol, position) in enumerate(positions.items()):
                value = position['size'] * position.get('current_price', position['entry_price'])
                weight = (value / total_value) * 100 if total_value > 0 else 0
                
                labels.append(symbol)
                values.append(weight)
                colors.append(color_palette[i % len(color_palette)])
            
            return jsonify({
                'labels': labels,
                'values': values,
                'colors': colors
            })
            
        except Exception as e:
            logger.error(f"Error getting portfolio allocation: {e}")
            return jsonify({'error': str(e)})
    
    def pnl_chart(self):
        """P&L chart data endpoint"""
        try:
            # Generate demo P&L data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            # Calculate cumulative P&L (demo data)
            daily_returns = np.random.normal(0.001, 0.02, 30)  # 0.1% daily return, 2% volatility
            cumulative_pnl = np.cumsum(daily_returns) * 100000  # Assume $100k portfolio
            
            return jsonify({
                'dates': [d.strftime('%Y-%m-%d') for d in dates],
                'pnl': cumulative_pnl.tolist()
            })
            
        except Exception as e:
            logger.error(f"Error getting P&L chart: {e}")
            return jsonify({'error': str(e)})
    
    def get_alerts(self):
        """Get current alerts endpoint"""
        try:
            active_alerts = self.portfolio_monitor.get_active_alerts()
            alert_summary = self.portfolio_monitor.get_alert_summary()
            
            alerts_data = []
            for alert in active_alerts[:10]:  # Latest 10 alerts
                alerts_data.append({
                    'id': alert.id,
                    'type': alert.type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged
                })
            
            return jsonify({
                'alerts': alerts_data,
                'summary': alert_summary
            })
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return jsonify({'error': str(e)})
    
    def acknowledge_alert(self, alert_id):
        """Acknowledge an alert endpoint"""
        try:
            self.portfolio_monitor.acknowledge_alert(alert_id)
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    def rebalancing_recommendations(self):
        """Get rebalancing recommendations endpoint"""
        try:
            positions = self.risk_manager.positions
            
            if not positions:
                return jsonify({'recommendations': []})
            
            # For demo, create simple target weights
            symbols = list(positions.keys())
            target_weights = {symbol: 1.0/len(symbols) for symbol in symbols}  # Equal weight
            
            recommendations = self.portfolio_analytics.analyze_rebalancing_needs(
                positions, target_weights, tolerance=0.05
            )
            
            recommendations_data = []
            for rec in recommendations:
                recommendations_data.append({
                    'symbol': rec.symbol,
                    'current_weight': rec.current_weight,
                    'target_weight': rec.target_weight,
                    'weight_deviation': rec.weight_deviation,
                    'action': rec.action,
                    'amount_to_trade': rec.amount_to_trade,
                    'urgency': rec.urgency,
                    'reason': rec.reason
                })
            
            return jsonify({'recommendations': recommendations_data})
            
        except Exception as e:
            logger.error(f"Error getting rebalancing recommendations: {e}")
            return jsonify({'error': str(e)})
    
    def risk_metrics(self):
        """Get risk metrics endpoint"""
        try:
            risk_data = self.risk_manager.calculate_risk_metrics()
            
            return jsonify({
                'current_drawdown': risk_data.current_drawdown,
                'max_drawdown': risk_data.max_drawdown,
                'risk_score': risk_data.risk_score,
                'sharpe_ratio': risk_data.sharpe_ratio,
                'position_count': len(self.risk_manager.positions),
                'capital_utilization': self.risk_manager.get_capital_utilization()
            })
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return jsonify({'error': str(e)})
    
    def _generate_demo_returns(self, days=30):
        """Generate demo portfolio returns"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = pd.Series(
            np.random.normal(0.001, 0.02, days),  # 0.1% mean return, 2% volatility
            index=dates
        )
        return returns
