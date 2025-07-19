"""
Portfolio Monitoring and Alerting System
Real-time monitoring of portfolio health and automated alerting
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable

from .modules.alert_manager import AlertManager, Alert, AlertSeverity, AlertType
from .modules.metrics_collector import MetricsCollector
from .modules.alert_checkers import (
    PerformanceAlertChecker,
    RiskAlertChecker,
    ConcentrationAlertChecker,
    RebalancingAlertChecker,
    VolatilityAlertChecker
)
from .modules.alert_handlers import LogAlertHandler, EmailAlertHandler, SlackAlertHandler

logger = logging.getLogger(__name__)


class PortfolioMonitor:
    """Real-time portfolio monitoring and alerting with modular architecture"""
    
    def __init__(self, 
                 alert_handlers: Optional[List[Callable]] = None,
                 check_interval: int = 60):
        self.check_interval = check_interval
        self.monitoring_active = False
        
        # Initialize components
        self.alert_manager = AlertManager()
        self.metrics_collector = MetricsCollector()
        
        # Initialize alert checkers
        self.checkers = [
            PerformanceAlertChecker(),
            RiskAlertChecker(),
            ConcentrationAlertChecker(),
            RebalancingAlertChecker(),
            VolatilityAlertChecker()
        ]
        
        # Set up alert handlers
        if alert_handlers:
            for handler in alert_handlers:
                self.alert_manager.add_handler(handler)
        
        # Monitoring state
        self.last_metrics = {}
        self.baseline_metrics = {}
        
        logger.info("PortfolioMonitor initialized with modular architecture")
    
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
            # Collect current metrics
            current_metrics = await self.metrics_collector.collect_metrics(
                self.risk_manager,
                self.portfolio_analytics,
                self.data_collector
            )
            
            if not current_metrics:
                return
            
            # Run all alert checkers
            for checker in self.checkers:
                alerts = await checker.check_alerts(
                    current_metrics,
                    self.last_metrics,
                    self.baseline_metrics
                )
                
                for alert in alerts:
                    await self.alert_manager.create_alert(alert)
            
            # Update tracking
            self.last_metrics = current_metrics
            
            # Clean up resolved alerts
            self.alert_manager.cleanup_resolved_alerts()
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        self.alert_manager.acknowledge_alert(alert_id)
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        self.alert_manager.resolve_alert(alert_id)
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get current active alerts"""
        return self.alert_manager.get_active_alerts(severity_filter)
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert status"""
        return self.alert_manager.get_alert_summary()
    
    def set_baseline_metrics(self, metrics: Dict):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics.copy()
        logger.info("Baseline metrics updated for monitoring")
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update alert thresholds"""
        for checker in self.checkers:
            checker.update_thresholds(new_thresholds)
        logger.info(f"Alert thresholds updated: {new_thresholds}")
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring_active = False
        logger.info("Portfolio monitoring stopped")

# Export alert handlers for convenience
__all__ = [
    'PortfolioMonitor',
    'LogAlertHandler',
    'EmailAlertHandler',
    'SlackAlertHandler',
    'Alert',
    'AlertSeverity',
    'AlertType'
]
