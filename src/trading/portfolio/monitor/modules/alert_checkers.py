"""
Alert checking modules
"""

import logging
from typing import Dict, List
from datetime import datetime
from abc import ABC, abstractmethod

from .alert_manager import Alert, AlertSeverity, AlertType

logger = logging.getLogger(__name__)


class AlertChecker(ABC):
    """Base class for alert checkers"""
    
    def __init__(self):
        self.thresholds = {}
    
    @abstractmethod
    async def check_alerts(self, current_metrics: Dict, 
                          last_metrics: Dict, 
                          baseline_metrics: Dict) -> List[Alert]:
        """Check for alerts based on metrics"""
        pass
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update thresholds for this checker"""
        self.thresholds.update(new_thresholds)
    
    def _create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                     title: str, message: str, data: Dict) -> Alert:
        """Create an alert object"""
        alert_id = f"{alert_type.value}_{severity.value}_{int(datetime.now().timestamp())}"
        return Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            data=data
        )


class PerformanceAlertChecker(AlertChecker):
    """Checks for performance-related alerts"""
    
    def __init__(self):
        super().__init__()
        self.thresholds = {
            'max_drawdown_warning': 0.10,
            'max_drawdown_critical': 0.20,
            'sharpe_ratio_warning': 0.0,
            'daily_loss_threshold': 0.05
        }
    
    async def check_alerts(self, current_metrics: Dict, 
                          last_metrics: Dict, 
                          baseline_metrics: Dict) -> List[Alert]:
        """Check for performance alerts"""
        alerts = []
        
        try:
            risk_metrics = current_metrics.get('risk_metrics')
            if not risk_metrics:
                return alerts
            
            # Drawdown alerts
            if hasattr(risk_metrics, 'current_drawdown'):
                current_drawdown = risk_metrics.current_drawdown
                
                if current_drawdown > self.thresholds['max_drawdown_critical']:
                    alerts.append(self._create_alert(
                        AlertType.DRAWDOWN,
                        AlertSeverity.CRITICAL,
                        "Critical Drawdown Level",
                        f"Portfolio drawdown reached {current_drawdown:.2%}, "
                        f"exceeding critical threshold of {self.thresholds['max_drawdown_critical']:.2%}",
                        {'drawdown': current_drawdown, 'threshold': self.thresholds['max_drawdown_critical']}
                    ))
                elif current_drawdown > self.thresholds['max_drawdown_warning']:
                    alerts.append(self._create_alert(
                        AlertType.DRAWDOWN,
                        AlertSeverity.HIGH,
                        "High Drawdown Warning",
                        f"Portfolio drawdown reached {current_drawdown:.2%}, "
                        f"approaching critical level",
                        {'drawdown': current_drawdown, 'threshold': self.thresholds['max_drawdown_warning']}
                    ))
            
            # Sharpe ratio alerts
            if hasattr(risk_metrics, 'sharpe_ratio') and risk_metrics.sharpe_ratio < self.thresholds['sharpe_ratio_warning']:
                alerts.append(self._create_alert(
                    AlertType.PERFORMANCE,
                    AlertSeverity.MEDIUM,
                    "Poor Risk-Adjusted Returns",
                    f"Sharpe ratio is {risk_metrics.sharpe_ratio:.2f}, indicating poor risk-adjusted performance",
                    {'sharpe_ratio': risk_metrics.sharpe_ratio}
                ))
            
            # Daily performance alerts
            daily_return = current_metrics.get('portfolio_return_today', 0)
            if daily_return < -self.thresholds['daily_loss_threshold']:
                alerts.append(self._create_alert(
                    AlertType.PERFORMANCE,
                    AlertSeverity.HIGH,
                    "Significant Daily Loss",
                    f"Portfolio down {daily_return:.2%} today",
                    {'daily_return': daily_return}
                ))
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
        
        return alerts


class RiskAlertChecker(AlertChecker):
    """Checks for risk-related alerts"""
    
    def __init__(self):
        super().__init__()
        self.thresholds = {
            'risk_score_extreme': 90,
            'risk_score_high': 75,
            'var_breach_multiplier': 2.0
        }
    
    async def check_alerts(self, current_metrics: Dict, 
                          last_metrics: Dict, 
                          baseline_metrics: Dict) -> List[Alert]:
        """Check for risk alerts"""
        alerts = []
        
        try:
            risk_metrics = current_metrics.get('risk_metrics')
            if not risk_metrics:
                return alerts
            
            # Risk score alerts
            if hasattr(risk_metrics, 'risk_score'):
                if risk_metrics.risk_score > self.thresholds['risk_score_extreme']:
                    alerts.append(self._create_alert(
                        AlertType.RISK,
                        AlertSeverity.CRITICAL,
                        "Extreme Risk Level",
                        f"Portfolio risk score reached {risk_metrics.risk_score}",
                        {'risk_score': risk_metrics.risk_score}
                    ))
                elif risk_metrics.risk_score > self.thresholds['risk_score_high']:
                    alerts.append(self._create_alert(
                        AlertType.RISK,
                        AlertSeverity.HIGH,
                        "High Risk Level",
                        f"Portfolio risk score is {risk_metrics.risk_score}",
                        {'risk_score': risk_metrics.risk_score}
                    ))
            
            # VaR breach alerts
            if hasattr(risk_metrics, 'var_95'):
                daily_return = current_metrics.get('portfolio_return_today', 0)
                if daily_return < risk_metrics.var_95 * self.thresholds['var_breach_multiplier']:
                    alerts.append(self._create_alert(
                        AlertType.RISK,
                        AlertSeverity.HIGH,
                        "VaR Breach",
                        f"Daily return {daily_return:.2%} exceeded VaR estimate",
                        {'daily_return': daily_return, 'var_95': risk_metrics.var_95}
                    ))
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
        
        return alerts


class ConcentrationAlertChecker(AlertChecker):
    """Checks for concentration risk alerts"""
    
    def __init__(self):
        super().__init__()
        self.thresholds = {
            'concentration_warning': 0.30,
            'concentration_critical': 0.50,
            'min_positions': 3
        }
    
    async def check_alerts(self, current_metrics: Dict, 
                          last_metrics: Dict, 
                          baseline_metrics: Dict) -> List[Alert]:
        """Check for concentration alerts"""
        alerts = []
        
        try:
            largest_weight = current_metrics.get('largest_position_weight', 0)
            
            if largest_weight > self.thresholds['concentration_critical']:
                alerts.append(self._create_alert(
                    AlertType.CONCENTRATION,
                    AlertSeverity.CRITICAL,
                    "Critical Concentration Risk",
                    f"Single position represents {largest_weight:.2%} of portfolio",
                    {'largest_weight': largest_weight, 'threshold': self.thresholds['concentration_critical']}
                ))
            elif largest_weight > self.thresholds['concentration_warning']:
                alerts.append(self._create_alert(
                    AlertType.CONCENTRATION,
                    AlertSeverity.MEDIUM,
                    "High Concentration Risk",
                    f"Single position represents {largest_weight:.2%} of portfolio",
                    {'largest_weight': largest_weight, 'threshold': self.thresholds['concentration_warning']}
                ))
            
            # Check for too few positions
            position_count = current_metrics.get('position_count', 0)
            if position_count < self.thresholds['min_positions']:
                alerts.append(self._create_alert(
                    AlertType.CONCENTRATION,
                    AlertSeverity.MEDIUM,
                    "Insufficient Diversification",
                    f"Portfolio has only {position_count} positions",
                    {'position_count': position_count}
                ))
            
        except Exception as e:
            logger.error(f"Error checking concentration alerts: {e}")
        
        return alerts


class RebalancingAlertChecker(AlertChecker):
    """Checks for rebalancing-related alerts"""
    
    def __init__(self):
        super().__init__()
        self.thresholds = {
            'rebalancing_drift_warning': 0.05,
            'rebalancing_drift_critical': 0.15
        }
    
    async def check_alerts(self, current_metrics: Dict, 
                          last_metrics: Dict, 
                          baseline_metrics: Dict) -> List[Alert]:
        """Check for rebalancing alerts"""
        alerts = []
        
        try:
            # Check for significant weight drift (if we have historical baseline)
            if baseline_metrics and 'weights' in baseline_metrics:
                weights = current_metrics.get('weights', {})
                baseline_weights = baseline_metrics['weights']
                
                large_drifts = []
                for symbol in weights:
                    if symbol in baseline_weights:
                        drift = abs(weights[symbol] - baseline_weights[symbol])
                        if drift > self.thresholds['rebalancing_drift_critical']:
                            large_drifts.append((symbol, drift))
                
                if large_drifts:
                    alerts.append(self._create_alert(
                        AlertType.REBALANCING,
                        AlertSeverity.HIGH,
                        "Significant Portfolio Drift",
                        f"Large weight changes detected: {large_drifts[:3]}",  # Show first 3
                        {'drifts': large_drifts}
                    ))
            
        except Exception as e:
            logger.error(f"Error checking rebalancing alerts: {e}")
        
        return alerts


class VolatilityAlertChecker(AlertChecker):
    """Checks for volatility-related alerts"""
    
    def __init__(self):
        super().__init__()
        self.thresholds = {
            'volatility_spike_multiplier': 2.0
        }
    
    async def check_alerts(self, current_metrics: Dict, 
                          last_metrics: Dict, 
                          baseline_metrics: Dict) -> List[Alert]:
        """Check for volatility alerts"""
        alerts = []
        
        try:
            # Calculate recent volatility (simplified)
            if last_metrics and 'portfolio_return_today' in last_metrics:
                recent_returns = [
                    current_metrics.get('portfolio_return_today', 0),
                    last_metrics.get('portfolio_return_today', 0)
                ]
                
                current_vol = abs(recent_returns[0] - recent_returns[1])
                
                # Compare with baseline if available
                if baseline_metrics and 'avg_volatility' in baseline_metrics:
                    baseline_vol = baseline_metrics['avg_volatility']
                    if current_vol > baseline_vol * self.thresholds['volatility_spike_multiplier']:
                        alerts.append(self._create_alert(
                            AlertType.VOLATILITY,
                            AlertSeverity.MEDIUM,
                            "Volatility Spike",
                            f"Current volatility {current_vol:.2%} is {current_vol/baseline_vol:.1f}x baseline",
                            {'current_volatility': current_vol, 'baseline_volatility': baseline_vol}
                        ))
            
        except Exception as e:
            logger.error(f"Error checking volatility alerts: {e}")
        
        return alerts
