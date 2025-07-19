"""
Dashboard Route Setup Module
Configures Flask routes and connects them to API handlers
"""

from flask import render_template
import logging

logger = logging.getLogger(__name__)


class RouteSetup:
    """Setup Flask routes for the dashboard"""
    
    def __init__(self, app, api_routes):
        """
        Initialize route setup
        
        Args:
            app: Flask application instance
            api_routes: APIRoutes instance with endpoint handlers
        """
        self.app = app
        self.api_routes = api_routes
    
    def setup_routes(self):
        """Configure all Flask routes"""
        
        # Main dashboard page
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        # API Routes
        @self.app.route('/api/portfolio/overview')
        def portfolio_overview():
            """Portfolio overview data"""
            return self.api_routes.portfolio_overview()
        
        @self.app.route('/api/portfolio/performance')
        def portfolio_performance():
            """Portfolio performance metrics"""
            return self.api_routes.portfolio_performance()
        
        @self.app.route('/api/portfolio/allocation')
        def portfolio_allocation():
            """Portfolio allocation chart data"""
            return self.api_routes.portfolio_allocation()
        
        @self.app.route('/api/portfolio/pnl_chart')
        def pnl_chart():
            """P&L chart data"""
            return self.api_routes.pnl_chart()
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get current alerts"""
            return self.api_routes.get_alerts()
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            """Acknowledge an alert"""
            return self.api_routes.acknowledge_alert(alert_id)
        
        @self.app.route('/api/rebalancing/recommendations')
        def rebalancing_recommendations():
            """Get rebalancing recommendations"""
            return self.api_routes.rebalancing_recommendations()
        
        @self.app.route('/api/risk/metrics')
        def risk_metrics():
            """Get risk metrics"""
            return self.api_routes.risk_metrics()
        
        logger.info("All dashboard routes configured successfully")
