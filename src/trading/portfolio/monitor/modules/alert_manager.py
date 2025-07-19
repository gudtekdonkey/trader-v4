"""
Alert management system
"""

import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

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


class AlertManager:
    """Manages portfolio alerts"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = []
        self.alert_handlers = []
    
    def add_handler(self, handler: Callable):
        """Add an alert handler"""
        self.alert_handlers.append(handler)
    
    async def create_alert(self, alert: Alert):
        """Create and process a new alert"""
        try:
            # Check if similar alert already exists
            existing_alert = self._find_similar_alert(alert.type, alert.severity)
            if existing_alert and not existing_alert.resolved:
                # Update existing alert instead of creating new one
                existing_alert.message = alert.message
                existing_alert.timestamp = datetime.now()
                existing_alert.data.update(alert.data)
                return
            
            # Store alert
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Process alert through handlers
            await self._process_alert(alert)
            
            logger.warning(f"Portfolio Alert [{alert.severity.value.upper()}]: {alert.title} - {alert.message}")
            
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
    
    def cleanup_resolved_alerts(self):
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
