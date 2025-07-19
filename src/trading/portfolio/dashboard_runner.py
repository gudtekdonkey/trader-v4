"""
Portfolio Dashboard Runner - Web dashboard launcher
Manages the Flask-based portfolio monitoring dashboard in a separate thread
for real-time visualization of portfolio metrics and alerts.

File: dashboard_runner.py
Modified: 2025-07-15
"""

import asyncio
import threading
import logging
from trading.portfolio.dashboard import PortfolioDashboard, create_dashboard_template

logger = logging.getLogger(__name__)

class DashboardManager:
    """Manages the portfolio dashboard in a separate thread"""
    
    def __init__(self, portfolio_analytics, portfolio_monitor, risk_manager, port=5000):
        self.dashboard = PortfolioDashboard(
            portfolio_analytics=portfolio_analytics,
            portfolio_monitor=portfolio_monitor,
            risk_manager=risk_manager,
            port=port
        )
        self.dashboard_thread = None
        self.running = False
    
    def start_dashboard(self):
        """Start the dashboard in a separate thread"""
        if self.running:
            logger.warning("Dashboard already running")
            return
        
        # Create template if it doesn't exist
        try:
            create_dashboard_template()
        except Exception as e:
            logger.error(f"Error creating dashboard template: {e}")
        
        def run_dashboard():
            try:
                logger.info("Starting portfolio dashboard...")
                self.dashboard.run(debug=False)
            except Exception as e:
                logger.error(f"Error running dashboard: {e}")
        
        self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        self.dashboard_thread.start()
        self.running = True
        
        logger.info(f"Portfolio dashboard started on http://localhost:{self.dashboard.port}")
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        self.running = False
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            logger.info("Stopping portfolio dashboard...")
            # Note: Flask doesn't have a clean shutdown in this setup
            # In production, would use proper WSGI server with shutdown capability

if __name__ == '__main__':
    # Demo mode - create mock objects for testing
    class MockRiskManager:
        def __init__(self):
            self.positions = {
                'BTC-USD': {
                    'side': 'long',
                    'size': 0.5,
                    'entry_price': 45000,
                    'current_price': 47000,
                    'unrealized_pnl': 1000,
                    'realized_pnl': 0
                },
                'ETH-USD': {
                    'side': 'long', 
                    'size': 3.0,
                    'entry_price': 3200,
                    'current_price': 3350,
                    'unrealized_pnl': 450,
                    'realized_pnl': 0
                }
            }
        
        def calculate_risk_metrics(self):
            class RiskMetrics:
                current_drawdown = 0.05
                max_drawdown = 0.08
                risk_score = 45
                sharpe_ratio = 1.2
            return RiskMetrics()
        
        def get_capital_utilization(self):
            return 0.75
    
    class MockPortfolioAnalytics:
        def calculate_portfolio_metrics(self, returns, positions=None):
            class Metrics:
                total_return = 0.15
                sharpe_ratio = 1.2
                sortino_ratio = 1.5
                max_drawdown = 0.08
                volatility = 0.18
                win_rate = 0.65
                profit_factor = 1.8
                var_95 = -0.03
                diversification_ratio = 0.85
            return Metrics()
        
        def analyze_rebalancing_needs(self, positions, target_weights, tolerance=0.05):
            return []  # No rebalancing needed in demo
    
    class MockPortfolioMonitor:
        def get_active_alerts(self):
            return []
        
        def get_alert_summary(self):
            return {
                'total_active': 0,
                'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                'by_type': {},
                'latest_alert': None
            }
        
        def acknowledge_alert(self, alert_id):
            pass
    
    # Create mock instances
    risk_manager = MockRiskManager()
    portfolio_analytics = MockPortfolioAnalytics() 
    portfolio_monitor = MockPortfolioMonitor()
    
    # Start dashboard
    dashboard_manager = DashboardManager(
        portfolio_analytics=portfolio_analytics,
        portfolio_monitor=portfolio_monitor,
        risk_manager=risk_manager,
        port=5000
    )
    
    dashboard_manager.start_dashboard()
    
    print("Dashboard running at http://localhost:5000")
    print("Press Ctrl+C to exit")
    
    try:
        # Keep the main thread alive
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        dashboard_manager.stop_dashboard()
