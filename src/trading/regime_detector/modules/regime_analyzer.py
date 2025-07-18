"""
Regime Analyzer Module

Handles regime analysis including statistics calculation,
transition predictions, and regime mapping.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


class RegimeAnalyzer:
    """
    Analyzes market regimes and provides statistical insights.
    
    This class calculates regime statistics, manages regime transitions,
    and provides predictive capabilities for future regime changes.
    """
    
    def __init__(self, n_regimes: int):
        """
        Initialize the regime analyzer.
        
        Args:
            n_regimes: Number of market regimes
        """
        self.n_regimes = n_regimes
    
    def calculate_regime_statistics(
        self, 
        data: pd.DataFrame, 
        regime_labels: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate statistical characteristics for each regime.
        
        Args:
            data: Market data
            regime_labels: Regime labels for each data point
            
        Returns:
            Dictionary of regime statistics
        """
        regime_stats = {}
        
        try:
            returns = data['close'].pct_change()
            
            for regime in range(self.n_regimes):
                regime_mask = regime_labels == regime
                
                if regime_mask.any():
                    regime_returns = returns[regime_mask]
                    
                    # Calculate statistics
                    stats = {
                        'mean_return': float(regime_returns.mean()),
                        'volatility': float(regime_returns.std()),
                        'frequency': float(regime_mask.sum() / len(regime_labels)),
                        'avg_duration': self._calculate_avg_duration(regime_labels, regime),
                        'sharpe_ratio': self._calculate_sharpe_ratio(regime_returns),
                        'max_drawdown': self._calculate_max_drawdown(data[regime_mask]),
                        'skewness': float(regime_returns.skew()),
                        'kurtosis': float(regime_returns.kurt()),
                        'win_rate': float((regime_returns > 0).sum() / len(regime_returns))
                    }
                    
                    regime_stats[regime] = stats
                else:
                    # No data for this regime
                    regime_stats[regime] = self._get_default_stats()
            
            return regime_stats
            
        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            # Return default stats for all regimes
            return {i: self._get_default_stats() for i in range(self.n_regimes)}
    
    def _get_default_stats(self) -> Dict[str, float]:
        """Get default statistics for regimes with no data."""
        return {
            'mean_return': 0.0,
            'volatility': 0.02,
            'frequency': 0.0,
            'avg_duration': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'win_rate': 0.5
        }
    
    def _calculate_avg_duration(
        self, 
        regime_sequence: np.ndarray, 
        regime: int
    ) -> float:
        """
        Calculate average duration of a regime.
        
        Args:
            regime_sequence: Sequence of regime labels
            regime: Regime to calculate duration for
            
        Returns:
            Average duration in periods
        """
        try:
            durations = []
            current_duration = 0
            
            for r in regime_sequence:
                if r == regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
            
            return float(np.mean(durations)) if durations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average duration: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio for a series of returns.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        try:
            excess_returns = returns - risk_free_rate
            if returns.std() > 0:
                sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
                return float(sharpe)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        """
        Calculate maximum drawdown for a price series.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Maximum drawdown as a percentage
        """
        try:
            if 'close' not in data.columns or len(data) == 0:
                return 0.0
            
            prices = data['close']
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            
            return float(drawdown.min())
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def create_regime_mapping(
        self, 
        regime_stats: Dict[int, Dict[str, float]]
    ) -> Dict[int, int]:
        """
        Create mapping from old to new regime numbers based on volatility.
        
        Args:
            regime_stats: Statistics for each regime
            
        Returns:
            Mapping dictionary
        """
        try:
            # Sort regimes by volatility (0 = lowest vol, n-1 = highest vol)
            sorted_regimes = sorted(
                regime_stats.items(), 
                key=lambda x: x[1]['volatility']
            )
            
            # Create mapping from old to new regime numbers
            regime_mapping = {
                old: new for new, (old, _) in enumerate(sorted_regimes)
            }
            
            return regime_mapping
            
        except Exception as e:
            logger.error(f"Error creating regime mapping: {e}")
            # Return identity mapping
            return {i: i for i in range(self.n_regimes)}
    
    def predict_next_regime(
        self, 
        current_regime: int, 
        steps: int,
        transition_matrix: np.ndarray
    ) -> Tuple[int, float]:
        """
        Predict the next regime and confidence.
        
        Args:
            current_regime: Current regime number
            steps: Number of steps ahead to predict
            transition_matrix: Regime transition probabilities
            
        Returns:
            Tuple of (most likely regime, confidence)
        """
        try:
            # Validate inputs
            if not 0 <= current_regime < self.n_regimes:
                logger.warning(f"Invalid current regime {current_regime}")
                return 1, 0.5  # Default to normal regime
            
            if steps <= 0:
                logger.warning(f"Invalid steps {steps}")
                return current_regime, 1.0
            
            # Current regime probability vector
            current_probs = np.zeros(self.n_regimes)
            current_probs[current_regime] = 1.0
            
            # Propagate probabilities
            for _ in range(steps):
                current_probs = current_probs @ transition_matrix
            
            # Most likely next regime
            next_regime = int(np.argmax(current_probs))
            confidence = float(current_probs[next_regime])
            
            return next_regime, confidence
            
        except Exception as e:
            logger.error(f"Error predicting next regime: {e}")
            return current_regime, 0.5
    
    def analyze_regime_transitions(
        self, 
        regime_history: List[Dict]
    ) -> Dict[str, any]:
        """
        Analyze regime transition patterns.
        
        Args:
            regime_history: Historical regime data
            
        Returns:
            Analysis of regime transitions
        """
        try:
            if len(regime_history) < 2:
                return {'message': 'Insufficient history for analysis'}
            
            transitions = []
            
            for i in range(1, len(regime_history)):
                if regime_history[i]['regime'] != regime_history[i-1]['regime']:
                    transitions.append({
                        'from': regime_history[i-1]['regime'],
                        'to': regime_history[i]['regime'],
                        'timestamp': regime_history[i]['timestamp']
                    })
            
            # Calculate transition frequencies
            transition_counts = {}
            for trans in transitions:
                key = (trans['from'], trans['to'])
                transition_counts[key] = transition_counts.get(key, 0) + 1
            
            # Most common transitions
            most_common = sorted(
                transition_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            return {
                'total_transitions': len(transitions),
                'transition_rate': len(transitions) / len(regime_history),
                'most_common_transitions': most_common,
                'recent_transitions': transitions[-5:] if transitions else []
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regime transitions: {e}")
            return {'error': str(e)}
