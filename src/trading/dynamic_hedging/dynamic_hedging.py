"""
File: dynamic_hedging.py
Modified: 2025-07-18
Changes Summary:
- Modularized into separate components for better maintainability
- Main hedging system coordination remains here
- Components moved to modules/ directory
"""

"""
Dynamic Hedging System Module

Implements portfolio hedging strategies to manage downside risk.
This module provides automated hedging recommendations and execution
based on portfolio exposure, market conditions, and risk metrics.

Classes:
    DynamicHedgingSystem: Main hedging system implementation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import traceback

# Import modularized components
from .modules.hedge_types import HedgeRecommendation
from .modules.exposure_calculator import ExposureCalculator
from .modules.hedge_analyzer import HedgeAnalyzer
from .modules.hedge_executor import HedgeExecutor
from .modules.hedge_position_manager import HedgePositionManager
from .modules.hedge_instruments import HedgeInstrumentManager

logger = logging.getLogger(__name__)


class DynamicHedgingSystem:
    """
    Dynamic portfolio hedging system with comprehensive error handling.
    
    This system analyzes portfolio exposure and market conditions to recommend
    and execute hedging strategies. It supports various hedge types including
    beta hedging, volatility hedging, correlation hedging, and tail risk protection.
    
    Attributes:
        hedge_positions: Currently active hedge positions
        hedge_history: Historical hedge positions and their outcomes
        hedge_thresholds: Thresholds for triggering different hedge types
        hedge_instruments: Available hedging instruments and their characteristics
    """
    
    def __init__(self) -> None:
        """Initialize the Dynamic Hedging System with error handling."""
        try:
            # Initialize components
            self.exposure_calculator = ExposureCalculator()
            self.hedge_analyzer = HedgeAnalyzer()
            self.hedge_executor = HedgeExecutor()
            self.position_manager = HedgePositionManager()
            self.instrument_manager = HedgeInstrumentManager()
            
            # Hedging parameters with validation
            self.hedge_thresholds: Dict[str, float] = {
                'portfolio_beta_threshold': 1.2,  # Hedge when portfolio beta > 1.2
                'correlation_threshold': 0.8,     # Hedge when correlation > 0.8
                'volatility_threshold': 0.25,     # Hedge when volatility > 25%
                'drawdown_threshold': 0.08,       # Hedge when drawdown > 8%
                'var_threshold': 0.05             # Hedge when VaR > 5%
            }
            
            # Validate thresholds
            for key, value in self.hedge_thresholds.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    logger.warning(f"Invalid threshold {key}: {value}, using default")
                    self.hedge_thresholds[key] = 0.1
            
            # Error tracking
            self.error_count = 0
            self.max_errors = 50
            
            logger.info("DynamicHedgingSystem initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DynamicHedgingSystem: {e}")
            raise
    
    def analyze_hedge_needs(
        self, 
        portfolio_positions: Dict[str, Dict],
        market_data: Dict[str, any],
        risk_metrics: Dict[str, float]
    ) -> List[HedgeRecommendation]:
        """
        Analyze portfolio and recommend hedging strategies with error handling.
        
        This method evaluates the portfolio across multiple dimensions and
        generates hedge recommendations based on current risk exposures.
        
        Args:
            portfolio_positions: Current portfolio positions
            market_data: Market data for all assets
            risk_metrics: Portfolio risk metrics (VaR, drawdown, etc.)
            
        Returns:
            List of hedge recommendations prioritized by urgency
        """
        try:
            # Validate inputs
            if not portfolio_positions:
                logger.warning("No portfolio positions provided for hedge analysis")
                return []
            
            if not isinstance(risk_metrics, dict):
                logger.error("Invalid risk_metrics format")
                return []
            
            recommendations = []
            
            # Calculate portfolio exposure metrics
            exposure_metrics = self.exposure_calculator.calculate_exposure_metrics(
                portfolio_positions, 
                market_data
            )
            
            if not exposure_metrics:
                logger.warning("Failed to calculate exposure metrics")
                return []
            
            # Check for beta hedging needs
            beta_hedge = self.hedge_analyzer.check_beta_hedge(
                exposure_metrics, 
                risk_metrics,
                self.hedge_thresholds
            )
            if beta_hedge:
                recommendations.append(beta_hedge)
            
            # Check for correlation hedging needs
            correlation_hedge = self.hedge_analyzer.check_correlation_hedge(
                exposure_metrics,
                market_data,
                self.hedge_thresholds
            )
            if correlation_hedge:
                recommendations.append(correlation_hedge)
            
            # Check for volatility hedging needs
            volatility_hedge = self.hedge_analyzer.check_volatility_hedge(
                exposure_metrics,
                risk_metrics,
                self.hedge_thresholds
            )
            if volatility_hedge:
                recommendations.append(volatility_hedge)
            
            # Check for tail risk hedging
            tail_risk_hedge = self.hedge_analyzer.check_tail_risk_hedge(
                exposure_metrics,
                risk_metrics,
                self.hedge_thresholds
            )
            if tail_risk_hedge:
                recommendations.append(tail_risk_hedge)
            
            # Check for sector/theme concentration hedging
            concentration_hedge = self.hedge_analyzer.check_concentration_hedge(
                exposure_metrics,
                self.hedge_thresholds
            )
            if concentration_hedge:
                recommendations.append(concentration_hedge)
            
            # Check error threshold
            if self.error_count > self.max_errors:
                logger.critical(f"Too many errors in hedge analysis: {self.error_count}")
                return []
            
            # Prioritize recommendations
            recommendations = self._prioritize_recommendations(recommendations)
            
            logger.info(f"Generated {len(recommendations)} hedge recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Critical error analyzing hedge needs: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _prioritize_recommendations(
        self,
        recommendations: List[HedgeRecommendation]
    ) -> List[HedgeRecommendation]:
        """
        Prioritize hedge recommendations by urgency and impact.
        
        Args:
            recommendations: List of hedge recommendations
            
        Returns:
            Sorted list of recommendations
        """
        try:
            # Define urgency weights
            urgency_weights = {
                'high': 3,
                'medium': 2,
                'low': 1
            }
            
            # Sort by urgency and expected protection
            return sorted(
                recommendations,
                key=lambda r: (
                    urgency_weights.get(r.urgency, 1),
                    r.expected_protection
                ),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error prioritizing recommendations: {e}")
            return recommendations
    
    async def execute_hedge(
        self, 
        recommendation: HedgeRecommendation, 
        order_executor
    ) -> Dict[str, any]:
        """
        Execute a hedge recommendation with error handling.
        
        Args:
            recommendation: Hedge recommendation to execute
            order_executor: Order execution interface
            
        Returns:
            Execution result dictionary
        """
        try:
            # Validate recommendation
            if not recommendation or not isinstance(recommendation, HedgeRecommendation):
                logger.error("Invalid hedge recommendation")
                return {'status': 'error', 'error': 'Invalid recommendation'}
            
            # Execute hedge through the hedge executor
            result = await self.hedge_executor.execute_hedge(
                recommendation,
                order_executor
            )
            
            if result.get('status') == 'success':
                # Register hedge position
                hedge_position = self.position_manager.register_hedge_position(
                    result['hedge_id'],
                    recommendation,
                    result['hedge_position']
                )
                
                logger.info(f"Hedge executed successfully: {hedge_position['hedge_id']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error executing hedge: {e}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_hedge_positions(self, current_prices: Dict[str, float]) -> None:
        """
        Update hedge position values and P&L with error handling.
        
        Args:
            current_prices: Current market prices
        """
        try:
            # Validate input
            if not isinstance(current_prices, dict):
                logger.error("Invalid current_prices format")
                return
            
            # Update positions through position manager
            self.position_manager.update_positions(current_prices)
            
            # Check for hedge adjustments needed
            adjustments = self.hedge_analyzer.check_hedge_adjustments(
                self.position_manager.get_active_positions(),
                current_prices
            )
            
            if adjustments:
                logger.info(f"Hedge adjustments needed: {len(adjustments)}")
                # Could trigger automatic adjustments here
                
        except Exception as e:
            logger.error(f"Critical error updating hedge positions: {e}")
            logger.error(traceback.format_exc())
    
    def get_hedge_portfolio_summary(self) -> Dict[str, any]:
        """
        Get summary of current hedging portfolio.
        
        Returns:
            Summary dictionary with hedge positions and performance
        """
        try:
            active_positions = self.position_manager.get_active_positions()
            
            if not active_positions:
                return {
                    'message': 'No active hedge positions',
                    'total_protection': 0,
                    'total_cost': 0
                }
            
            # Calculate aggregate metrics
            total_protection = sum(
                pos.get('expected_protection', 0) 
                for pos in active_positions.values()
            )
            
            total_cost = sum(
                pos.get('expected_cost', 0) 
                for pos in active_positions.values()
            )
            
            total_pnl = sum(
                pos.get('unrealized_pnl', 0) 
                for pos in active_positions.values()
            )
            
            return {
                'active_hedges': len(active_positions),
                'total_protection': total_protection,
                'total_cost': total_cost,
                'total_pnl': total_pnl,
                'efficiency': total_protection / total_cost if total_cost > 0 else 0,
                'positions': list(active_positions.values()),
                'instruments': self.instrument_manager.get_available_instruments()
            }
            
        except Exception as e:
            logger.error(f"Error getting hedge portfolio summary: {e}")
            return {'error': str(e)}
    
    def close_hedge_position(
        self,
        hedge_id: str,
        reason: str = 'manual'
    ) -> Dict[str, any]:
        """
        Close a hedge position.
        
        Args:
            hedge_id: Hedge position ID
            reason: Reason for closing
            
        Returns:
            Closing result
        """
        try:
            return self.position_manager.close_position(hedge_id, reason)
        except Exception as e:
            logger.error(f"Error closing hedge position: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_hedge_analytics(self) -> Dict[str, any]:
        """
        Get analytics on hedging performance.
        
        Returns:
            Analytics dictionary
        """
        try:
            return {
                'position_analytics': self.position_manager.get_analytics(),
                'instrument_usage': self.instrument_manager.get_usage_statistics(),
                'hedge_effectiveness': self.hedge_analyzer.analyze_effectiveness(
                    self.position_manager.get_historical_positions()
                ),
                'cost_analysis': self.hedge_analyzer.analyze_costs(
                    self.position_manager.get_historical_positions()
                )
            }
        except Exception as e:
            logger.error(f"Error getting hedge analytics: {e}")
            return {'error': str(e)}

"""
MODULARIZATION_SUMMARY:
- Original file: 800+ lines
- Main file: ~300 lines (core coordination)
- Modules created:
  - hedge_types.py: Hedge data structures and types
  - exposure_calculator.py: Portfolio exposure calculations
  - hedge_analyzer.py: Hedge analysis and recommendations
  - hedge_executor.py: Hedge execution logic
  - hedge_position_manager.py: Hedge position tracking
  - hedge_instruments.py: Hedging instrument management
- Benefits:
  - Clearer separation of concerns
  - Easier testing of hedge strategies
  - Better position tracking
  - Modular analytics
"""
