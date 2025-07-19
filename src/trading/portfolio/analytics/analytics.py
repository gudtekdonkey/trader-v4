"""
Portfolio Analytics Module
Provides advanced portfolio analysis, performance metrics, and rebalancing insights
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime

from .modules.metrics_calculator import MetricsCalculator
from .modules.rebalancing_analyzer import RebalancingAnalyzer
from .modules.risk_analyzer import RiskAnalyzer
from .modules.regime_detector import RegimeDetector
from .modules.report_generator import ReportGenerator
from .modules.data_types import PortfolioMetrics, RebalancingRecommendation

logger = logging.getLogger(__name__)


class PortfolioAnalytics:
    """Advanced portfolio analytics and optimization with modular architecture"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
        # Initialize modules
        self.metrics_calculator = MetricsCalculator(risk_free_rate)
        self.rebalancing_analyzer = RebalancingAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.regime_detector = RegimeDetector()
        self.report_generator = ReportGenerator()
        
        # Performance tracking
        self.performance_history = []
        
        logger.info("PortfolioAnalytics initialized with modular architecture")
    
    def calculate_portfolio_metrics(self, 
                                  returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  positions: Dict = None) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        return self.metrics_calculator.calculate_metrics(returns, benchmark_returns, positions)
    
    def analyze_rebalancing_needs(self,
                                current_positions: Dict,
                                target_weights: Dict,
                                tolerance: float = 0.05) -> List[RebalancingRecommendation]:
        """Analyze portfolio and generate rebalancing recommendations"""
        return self.rebalancing_analyzer.analyze_rebalancing_needs(
            current_positions, target_weights, tolerance
        )
    
    def calculate_portfolio_risk_metrics(self, 
                                       returns_matrix: pd.DataFrame,
                                       weights) -> Dict:
        """Calculate advanced portfolio risk metrics"""
        return self.risk_analyzer.calculate_risk_metrics(returns_matrix, weights)
    
    def detect_regime_changes(self, 
                            returns: pd.Series, 
                            window: int = 60) -> Dict:
        """Detect market regime changes that might affect portfolio allocation"""
        return self.regime_detector.detect_regime_changes(returns, window)
    
    def generate_portfolio_report(self,
                                positions: Dict,
                                returns: pd.Series,
                                target_weights: Dict = None) -> Dict:
        """Generate comprehensive portfolio analysis report"""
        # Gather data from all modules
        metrics = None
        if len(returns) > 0:
            metrics = self.calculate_portfolio_metrics(returns, positions=positions)
        
        risk_metrics = None
        if len(positions) > 1 and len(returns) >= 30:
            # Create returns matrix (simplified)
            symbols = list(positions.keys())
            weights = []
            for symbol in symbols:
                pos = positions[symbol]
                value = pos['size'] * pos.get('current_price', pos['entry_price'])
                weights.append(value)
            
            total_value = sum(weights)
            if total_value > 0:
                weights = [w / total_value for w in weights]
                returns_matrix = pd.DataFrame({
                    symbol: returns for symbol in symbols
                })
                risk_metrics = self.calculate_portfolio_risk_metrics(returns_matrix, weights)
        
        rebalancing_recommendations = None
        if target_weights:
            rebalancing_recommendations = self.analyze_rebalancing_needs(
                positions, target_weights
            )
        
        regime_info = None
        if len(returns) >= 60:
            regime_info = self.detect_regime_changes(returns)
        
        # Generate report
        return self.report_generator.generate_report(
            positions, metrics, risk_metrics, 
            rebalancing_recommendations, regime_info
        )
    
    def optimize_rebalancing_timing(self,
                                  recommendations: List[RebalancingRecommendation],
                                  market_conditions: Dict) -> List[Dict]:
        """Optimize the timing and execution of rebalancing trades"""
        return self.rebalancing_analyzer.optimize_timing(recommendations, market_conditions)
    
    def track_rebalancing_performance(self,
                                    executed_trades: List[Dict],
                                    original_plan: List[Dict]) -> Dict:
        """Track and analyze rebalancing execution performance"""
        analysis = self.rebalancing_analyzer.track_performance(executed_trades, original_plan)
        
        # Store in history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'analysis': analysis
        })
        
        return analysis
