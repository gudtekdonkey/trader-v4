"""
Portfolio risk analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Analyzes portfolio risk metrics"""
    
    def calculate_risk_metrics(self, 
                              returns_matrix: pd.DataFrame,
                              weights: np.ndarray) -> Dict:
        """Calculate advanced portfolio risk metrics"""
        try:
            # Portfolio returns
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            
            # Correlation matrix
            correlation_matrix = returns_matrix.corr()
            
            # Portfolio volatility
            portfolio_variance = np.dot(weights.T, np.dot(returns_matrix.cov() * 252, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Component contributions to risk
            marginal_risk = np.dot(returns_matrix.cov() * 252, weights) / portfolio_volatility
            component_risk = weights * marginal_risk
            risk_contribution = component_risk / portfolio_volatility
            
            # Diversification metrics
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            effective_number_assets = 1 / (weights**2).sum()
            
            # Maximum component risk
            max_component_risk = risk_contribution.max()
            
            # Risk concentration
            risk_concentration = (risk_contribution**2).sum()
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'component_risk_contributions': dict(zip(returns_matrix.columns, risk_contribution)),
                'marginal_risk_contributions': dict(zip(returns_matrix.columns, marginal_risk)),
                'average_correlation': avg_correlation,
                'effective_number_assets': effective_number_assets,
                'max_component_risk': max_component_risk,
                'risk_concentration': risk_concentration,
                'correlation_matrix': correlation_matrix.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}
