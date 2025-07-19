"""
Portfolio Dashboard Web Interface
Real-time portfolio monitoring and analytics visualization
"""

from flask import Flask
import logging
import os
from typing import Optional

from .modules.api_routes import APIRoutes
from .modules.route_setup import RouteSetup
from .modules.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class PortfolioDashboard:
    """Web dashboard for portfolio monitoring"""
    
    def __init__(self, portfolio_analytics, portfolio_monitor, risk_manager, port=5000):
        """
        Initialize portfolio dashboard
        
        Args:
            portfolio_analytics: Portfolio analytics instance
            portfolio_monitor: Portfolio monitor instance
            risk_manager: Risk manager instance
            port: Port to run dashboard on
        """
        # Set template folder relative to project root
        template_dir = os.path.join(
            os.path.dirname(__file__), 
            '../../../templates'
        )
        
        self.app = Flask(__name__, template_folder=template_dir)
        self.portfolio_analytics = portfolio_analytics
        self.portfolio_monitor = portfolio_monitor
        self.risk_manager = risk_manager
        self.port = port
        
        # Initialize modules
        self.api_routes = APIRoutes(
            portfolio_analytics,
            portfolio_monitor,
            risk_manager
        )
        
        self.template_manager = TemplateManager()
        
        # Setup routes
        self.route_setup = RouteSetup(self.app, self.api_routes)
        self.route_setup.setup_routes()
        
        logger.info("Portfolio dashboard initialized")
    
    def run(self, debug: bool = False, host: str = '0.0.0.0') -> None:
        """
        Run the dashboard server
        
        Args:
            debug: Enable debug mode
            host: Host to bind to
        """
        logger.info(f"Starting portfolio dashboard on {host}:{self.port}")
        
        # Ensure templates exist
        self.template_manager.ensure_templates_exist()
        
        # Run Flask app
        self.app.run(host=host, port=self.port, debug=debug)
    
    def get_app(self) -> Flask:
        """
        Get Flask app instance (for testing or WSGI deployment)
        
        Returns:
            Flask app instance
        """
        return self.app
    
    def update_port(self, port: int) -> None:
        """
        Update the port number
        
        Args:
            port: New port number
        """
        self.port = port
        logger.info(f"Dashboard port updated to {port}")


# Convenience function for creating dashboard instance
def create_dashboard(
    portfolio_analytics,
    portfolio_monitor, 
    risk_manager,
    port: int = 5000
) -> PortfolioDashboard:
    """
    Create and return a portfolio dashboard instance
    
    Args:
        portfolio_analytics: Portfolio analytics instance
        portfolio_monitor: Portfolio monitor instance
        risk_manager: Risk manager instance
        port: Port to run dashboard on
        
    Returns:
        PortfolioDashboard instance
    """
    return PortfolioDashboard(
        portfolio_analytics,
        portfolio_monitor,
        risk_manager,
        port
    )


# Template creation function for backward compatibility
def create_dashboard_template():
    """Create the dashboard HTML template file"""
    template_manager = TemplateManager()
    template_manager.create_dashboard_template()
    print("Dashboard template created at templates/dashboard.html")


if __name__ == '__main__':
    # If run directly, create template
    create_dashboard_template()
