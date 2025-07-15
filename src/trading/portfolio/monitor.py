"""
Portfolio Monitoring and Alerting System
Real-time monitoring of portfolio health and automated alerting
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    PERFORMANCE = "performance"
    RISK = "risk"
    REBALANCING = "rebalancing"
    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    SYSTEM = "system"

@dataclass
class Alert:
    """Portfolio alert structure"""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    data: Dict
    acknowledged: bool = False
    resolved: bool = False

class PortfolioMonitor:
    """Real-time portfolio monitoring and alerting"""
    
    def __init__(self, 
                 alert_handlers: Optional[List[Callable]] = None,
                 check_interval: int = 60):
        self.alert_handlers = alert_handlers or []
        self.check_interval = check_interval
        self.active_alerts = {}
        self.alert_history = []
        self.monitoring_active = False
        
        # Alert thresholds (configurable)
        self.thresholds = {
            'max_drawdown_warning': 0.10,
            'max_drawdown_critical': 0.20,
            'concentration_warning': 0.30,
            'concentration_critical': 0.50,
            'volatility_spike_multiplier': 2.0,
            'sharpe_ratio_warning': 0.0,
            'var_breach_multiplier': 2.0,
            'rebalancing_drift_warning': 0.05,
            'rebalancing_drift_critical': 0.15
        }
        
        # Monitoring state
        self.last_metrics = {}
        self.baseline_metrics = {}
        
    async def start_monitoring(self, 
                             risk_manager,
                             portfolio_analytics,
                             data_collector):
        """Start the portfolio monitoring loop"""
        self.monitoring_active = True
        self.risk_manager = risk_manager
        self.portfolio_analytics = portfolio_analytics
        self.data_collector = data_collector
        
        logger.info("Starting portfolio monitoring...")
        
        while self.monitoring_active:
            try:
                await self._run_monitoring_cycle()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _run_monitoring_cycle(self):
        """Run a single monitoring cycle"""
        try:
            # Get current portfolio state
            current_metrics = await self._collect_current_metrics()
            
            if not current_metrics:
                return
            
            # Run all monitoring checks
            await self._check_performance_alerts(current_metrics)
            await self._check_risk_alerts(current_metrics)
            await self._check_concentration_alerts(current_metrics)
            await self._check_rebalancing_alerts(current_metrics)
            await self._check_volatility_alerts(current_metrics)
            
            # Update tracking
            self.last_metrics = current_metrics
            
            # Clean up resolved alerts
            self._cleanup_resolved_alerts()
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
    
    async def _collect_current_metrics(self) -> Dict:
        """Collect current portfolio metrics for monitoring"""
        try:
            # Get positions from risk manager
            positions = self.risk_manager.positions
            
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
            risk_metrics = self.risk_manager.calculate_risk_metrics()
            
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
    
    async def _check_performance_alerts(self, metrics: Dict):
        """Check for performance-related alerts"""
        try:
            risk_metrics = metrics.get('risk_metrics')
            if not risk_metrics:
                return
            
            # Drawdown alerts
            current_drawdown = risk_metrics.current_drawdown
            
            if current_drawdown > self.thresholds['max_drawdown_critical']:
                await self._create_alert(
                    AlertType.DRAWDOWN,
                    AlertSeverity.CRITICAL,
                    "Critical Drawdown Level",
                    f"Portfolio drawdown reached {current_drawdown:.2%}, "
                    f"exceeding critical threshold of {self.thresholds['max_drawdown_critical']:.2%}",
                    {'drawdown': current_drawdown, 'threshold': self.thresholds['max_drawdown_critical']}
                )
            elif current_drawdown > self.thresholds['max_drawdown_warning']:
                await self._create_alert(
                    AlertType.DRAWDOWN,
                    AlertSeverity.HIGH,
                    "High Drawdown Warning",
                    f"Portfolio drawdown reached {current_drawdown:.2%}, "
                    f"approaching critical level",
                    {'drawdown': current_drawdown, 'threshold': self.thresholds['max_drawdown_warning']}
                )
            
            # Sharpe ratio alerts
            if hasattr(risk_metrics, 'sharpe_ratio') and risk_metrics.sharpe_ratio < self.thresholds['sharpe_ratio_warning']:
                await self._create_alert(
                    AlertType.PERFORMANCE,
                    AlertSeverity.MEDIUM,
                    "Poor Risk-Adjusted Returns",
                    f"Sharpe ratio is {risk_metrics.sharpe_ratio:.2f}, indicating poor risk-adjusted performance",
                    {'sharpe_ratio': risk_metrics.sharpe_ratio}
                )
            
            # Daily performance alerts
            daily_return = metrics.get('portfolio_return_today', 0)
            if daily_return < -0.05:  # -5% daily loss
                await self._create_alert(
                    AlertType.PERFORMANCE,
                    AlertSeverity.HIGH,
                    "Significant Daily Loss",
                    f"Portfolio down {daily_return:.2%} today",
                    {'daily_return': daily_return}
                )
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    async def _check_risk_alerts(self, metrics: Dict):
        """Check for risk-related alerts"""
        try:
            risk_metrics = metrics.get('risk_metrics')
            if not risk_metrics:
                return
            
            # Risk score alerts
            if hasattr(risk_metrics, 'risk_score'):
                if risk_metrics.risk_score > 90:
                    await self._create_alert(
                        AlertType.RISK,
                        AlertSeverity.CRITICAL,
                        "Extreme Risk Level",
                        f"Portfolio risk score reached {risk_metrics.risk_score}",
                        {'risk_score': risk_metrics.risk_score}
                    )
                elif risk_metrics.risk_score > 75:
                    await self._create_alert(
                        AlertType.RISK,
                        AlertSeverity.HIGH,
                        "High Risk Level",
                        f"Portfolio risk score is {risk_metrics.risk_score}",
                        {'risk_score': risk_metrics.risk_score}
                    )
            
            # VaR breach alerts (if available)
            if hasattr(risk_metrics, 'var_95'):
                daily_return = metrics.get('portfolio_return_today', 0)
                if daily_return < risk_metrics.var_95 * self.thresholds['var_breach_multiplier']:
                    await self._create_alert(
                        AlertType.RISK,
                        AlertSeverity.HIGH,
                        "VaR Breach",
                        f"Daily return {daily_return:.2%} exceeded VaR estimate",
                        {'daily_return': daily_return, 'var_95': risk_metrics.var_95}
                    )
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
    
    async def _check_concentration_alerts(self, metrics: Dict):
        """Check for concentration risk alerts"""
        try:
            largest_weight = metrics.get('largest_position_weight', 0)
            
            if largest_weight > self.thresholds['concentration_critical']:
                await self._create_alert(
                    AlertType.CONCENTRATION,
                    AlertSeverity.CRITICAL,
                    "Critical Concentration Risk",
                    f"Single position represents {largest_weight:.2%} of portfolio",
                    {'largest_weight': largest_weight, 'threshold': self.thresholds['concentration_critical']}
                )
            elif largest_weight > self.thresholds['concentration_warning']:
                await self._create_alert(
                    AlertType.CONCENTRATION,
                    AlertSeverity.MEDIUM,
                    "High Concentration Risk",
                    f"Single position represents {largest_weight:.2%} of portfolio",
                    {'largest_weight': largest_weight, 'threshold': self.thresholds['concentration_warning']}
                )
            
            # Check for too few positions
            position_count = metrics.get('position_count', 0)
            if position_count < 3:
                await self._create_alert(
                    AlertType.CONCENTRATION,
                    AlertSeverity.MEDIUM,
                    "Insufficient Diversification",
                    f"Portfolio has only {position_count} positions",
                    {'position_count': position_count}
                )
            
        except Exception as e:
            logger.error(f"Error checking concentration alerts: {e}")
    
    async def _check_rebalancing_alerts(self, metrics: Dict):
        """Check for rebalancing-related alerts"""
        try:
            # This would require target weights - simplified for now
            weights = metrics.get('weights', {})
            
            # Check for significant weight drift (if we have historical baseline)
            if self.baseline_metrics and 'weights' in self.baseline_metrics:
                baseline_weights = self.baseline_metrics['weights']
                
                large_drifts = []
                for symbol in weights:
                    if symbol in baseline_weights:
                        drift = abs(weights[symbol] - baseline_weights[symbol])
                        if drift > self.thresholds['rebalancing_drift_critical']:
                            large_drifts.append((symbol, drift))
                
                if large_drifts:
                    await self._create_alert(
                        AlertType.REBALANCING,
                        AlertSeverity.HIGH,
                        "Significant Portfolio Drift",
                        f"Large weight changes detected: {large_drifts[:3]}",  # Show first 3
                        {'drifts': large_drifts}
                    )
            
        except Exception as e:
            logger.error(f"Error checking rebalancing alerts: {e}")
    
    async def _check_volatility_alerts(self, metrics: Dict):
        """Check for volatility-related alerts"""
        try:
            # Calculate recent volatility (simplified)
            if self.last_metrics and 'portfolio_return_today' in self.last_metrics:
                recent_returns = [
                    metrics.get('portfolio_return_today', 0),
                    self.last_metrics.get('portfolio_return_today', 0)
                ]
                
                current_vol = abs(recent_returns[0] - recent_returns[1])
                
                # Compare with baseline if available
                if self.baseline_metrics and 'avg_volatility' in self.baseline_metrics:
                    baseline_vol = self.baseline_metrics['avg_volatility']
                    if current_vol > baseline_vol * self.thresholds['volatility_spike_multiplier']:
                        await self._create_alert(
                            AlertType.VOLATILITY,
                            AlertSeverity.MEDIUM,
                            "Volatility Spike",
                            f"Current volatility {current_vol:.2%} is {current_vol/baseline_vol:.1f}x baseline",
                            {'current_volatility': current_vol, 'baseline_volatility': baseline_vol}
                        )
            
        except Exception as e:
            logger.error(f"Error checking volatility alerts: {e}")
    
    async def _create_alert(self, 
                          alert_type: AlertType, 
                          severity: AlertSeverity, 
                          title: str, 
                          message: str, 
                          data: Dict):
        """Create and process a new alert"""
        try:
            # Generate unique alert ID
            alert_id = f"{alert_type.value}_{severity.value}_{int(datetime.now().timestamp())}"
            
            # Check if similar alert already exists
            existing_alert = self._find_similar_alert(alert_type, severity)
            if existing_alert and not existing_alert.resolved:
                # Update existing alert instead of creating new one
                existing_alert.message = message
                existing_alert.timestamp = datetime.now()
                existing_alert.data.update(data)
                return
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                type=alert_type,
                severity=severity,
                title=title,
                message=message,
                timestamp=datetime.now(),
                data=data
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Process alert through handlers
            await self._process_alert(alert)
            
            logger.warning(f"Portfolio Alert [{severity.value.upper()}]: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    def _find_similar_alert(self, alert_type: AlertType, severity: AlertSeverity) -> Optional[Alert]:
        """Find existing similar alert"""
        for alert in self.active_alerts.values():
            if alert.type == alert_type and alert.severity == severity and not alert.resolved:
                return alert
        return None
    
    async def _process_alert(self, alert: Alert):
        """Process alert through all registered handlers"""
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        try:
            current_time = datetime.now()
            cleanup_threshold = timedelta(hours=24)  # Keep resolved alerts for 24 hours
            
            alerts_to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if alert.resolved and (current_time - alert.timestamp) > cleanup_threshold:
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
            
        except Exception as e:
            logger.error(f"Error cleaning up alerts: {e}")
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get current active alerts"""
        alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        return sorted(alerts, key=lambda x: (x.severity.value, x.timestamp), reverse=True)
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert status"""
        active_alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
        
        summary = {
            'total_active': len(active_alerts),
            'by_severity': {
                'critical': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                'high': len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                'medium': len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
                'low': len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
            },
            'by_type': {},
            'latest_alert': max(active_alerts, key=lambda x: x.timestamp) if active_alerts else None
        }
        
        # Count by type
        for alert_type in AlertType:
            summary['by_type'][alert_type.value] = len([
                a for a in active_alerts if a.type == alert_type
            ])
        
        return summary
    
    def set_baseline_metrics(self, metrics: Dict):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics.copy()
        logger.info("Baseline metrics updated for monitoring")
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update alert thresholds"""
        self.thresholds.update(new_thresholds)
        logger.info(f"Alert thresholds updated: {new_thresholds}")
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring_active = False
        logger.info("Portfolio monitoring stopped")

# Alert handler implementations
class LogAlertHandler:
    """Log alerts to file"""
    
    def __init__(self, log_file: str = "portfolio_alerts.log"):
        self.log_file = log_file
    
    async def __call__(self, alert: Alert):
        """Handle alert by logging"""
        try:
            log_entry = {
                'timestamp': alert.timestamp.isoformat(),
                'id': alert.id,
                'type': alert.type.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'data': alert.data
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error in log alert handler: {e}")

class EmailAlertHandler:
    """Send alerts via email (placeholder)"""
    
    def __init__(self, email_config: Dict):
        self.email_config = email_config
    
    async def __call__(self, alert: Alert):
        """Handle alert by sending email"""
        # This would implement actual email sending
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            logger.info(f"Would send email alert: {alert.title}")

class SlackAlertHandler:
    """Send alerts to Slack (placeholder)"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def __call__(self, alert: Alert):
        """Handle alert by sending to Slack"""
        # This would implement actual Slack webhook integration
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            logger.info(f"Would send Slack alert: {alert.title}")
