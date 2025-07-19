"""
Dashboard Template Manager Module
Manages HTML templates and static content
"""

import os
import logging

logger = logging.getLogger(__name__)

# Dashboard HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Trading Portfolio Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .alert-badge {
            position: absolute;
            top: -5px;
            right: -5px;
        }
        .portfolio-card {
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-chart-line"></i> Crypto Trading Portfolio Dashboard
            </span>
            <div class="position-relative">
                <button class="btn btn-outline-light" id="alertsBtn">
                    <i class="fas fa-bell"></i> Alerts
                    <span class="badge bg-danger alert-badge" id="alertCount">0</span>
                </button>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Overview Row -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <h6><i class="fas fa-wallet"></i> Portfolio Value</h6>
                    <h3 id="totalValue">$0</h3>
                    <small id="totalReturn" class="text-light">0.00%</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h6><i class="fas fa-chart-line"></i> Total P&L</h6>
                    <h3 id="totalPnl">$0</h3>
                    <small id="winRate" class="text-light">Win Rate: 0%</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h6><i class="fas fa-exclamation-triangle"></i> Risk Score</h6>
                    <h3 id="riskScore">0</h3>
                    <small id="drawdown" class="text-light">Drawdown: 0%</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h6><i class="fas fa-coins"></i> Positions</h6>
                    <h3 id="positionCount">0</h3>
                    <small id="sharpeRatio" class="text-light">Sharpe: 0.00</small>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row">
            <div class="col-md-8">
                <div class="chart-container">
                    <h5><i class="fas fa-chart-area"></i> Portfolio P&L</h5>
                    <div id="pnlChart"></div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="chart-container">
                    <h5><i class="fas fa-pie-chart"></i> Asset Allocation</h5>
                    <div id="allocationChart"></div>
                </div>
            </div>
        </div>

        <!-- Positions and Alerts Row -->
        <div class="row">
            <div class="col-md-8">
                <div class="portfolio-card card">
                    <div class="card-header">
                        <h5><i class="fas fa-list"></i> Current Positions</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover" id="positionsTable">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Side</th>
                                        <th>Size</th>
                                        <th>Entry Price</th>
                                        <th>Current Price</th>
                                        <th>P&L</th>
                                        <th>Weight</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="portfolio-card card">
                    <div class="card-header">
                        <h5><i class="fas fa-exclamation-circle"></i> Recent Alerts</h5>
                    </div>
                    <div class="card-body" id="alertsList" style="max-height: 400px; overflow-y: auto;">
                        <!-- Alerts will be loaded here -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Rebalancing Recommendations -->
        <div class="row">
            <div class="col-12">
                <div class="portfolio-card card">
                    <div class="card-header">
                        <h5><i class="fas fa-balance-scale"></i> Rebalancing Recommendations</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover" id="rebalancingTable">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Current Weight</th>
                                        <th>Target Weight</th>
                                        <th>Deviation</th>
                                        <th>Action</th>
                                        <th>Urgency</th>
                                        <th>Reason</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/js/dashboard.js"></script>
</body>
</html>
"""


class TemplateManager:
    """Manages dashboard templates and static content"""
    
    def __init__(self, template_dir=None):
        """
        Initialize template manager
        
        Args:
            template_dir: Directory for templates (optional)
        """
        if template_dir is None:
            # Default to templates directory relative to dashboard
            self.template_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                '../../templates'
            )
        else:
            self.template_dir = template_dir
        
        self.static_dir = os.path.join(
            os.path.dirname(self.template_dir),
            'static'
        )
    
    def create_dashboard_template(self):
        """Create the dashboard HTML template file"""
        try:
            # Create templates directory if it doesn't exist
            os.makedirs(self.template_dir, exist_ok=True)
            
            template_path = os.path.join(self.template_dir, 'dashboard.html')
            with open(template_path, 'w') as f:
                f.write(DASHBOARD_HTML)
            
            logger.info(f"Dashboard template created at {template_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dashboard template: {e}")
            return False
    
    def create_static_files(self, dashboard_js_content):
        """
        Create static JavaScript files
        
        Args:
            dashboard_js_content: JavaScript content for dashboard
        """
        try:
            # Create static directories
            js_dir = os.path.join(self.static_dir, 'js')
            css_dir = os.path.join(self.static_dir, 'css')
            
            os.makedirs(js_dir, exist_ok=True)
            os.makedirs(css_dir, exist_ok=True)
            
            # Write JavaScript file
            js_path = os.path.join(js_dir, 'dashboard.js')
            with open(js_path, 'w') as f:
                f.write(dashboard_js_content)
            
            logger.info(f"Static files created in {self.static_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating static files: {e}")
            return False
    
    def get_template_path(self):
        """Get the template directory path"""
        return self.template_dir
    
    def get_static_path(self):
        """Get the static files directory path"""
        return self.static_dir
    
    def template_exists(self, template_name='dashboard.html'):
        """Check if a template exists"""
        template_path = os.path.join(self.template_dir, template_name)
        return os.path.exists(template_path)
    
    def ensure_templates_exist(self):
        """Ensure all required templates exist"""
        if not self.template_exists():
            logger.warning("Dashboard template not found, creating...")
            return self.create_dashboard_template()
        return True
