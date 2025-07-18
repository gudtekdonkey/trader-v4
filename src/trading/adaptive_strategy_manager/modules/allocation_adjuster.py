"""
Allocation adjustment module for adaptive strategy management.
Handles dynamic adjustments based on market conditions and signals.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AllocationAdjuster:
    """Handles dynamic allocation adjustments"""
    
    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
        
    def adjust_for_performance(self, base_allocation: Dict[str, float],
                             performance_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust allocation based on recent strategy performance.
        
        Args:
            base_allocation: Base allocation weights by strategy
            performance_scores: Performance scores for each strategy
            
        Returns:
            Performance-adjusted allocation weights
        """
        try:
            adjusted = base_allocation.copy()
            
            # Check if we have valid performance scores
            if not performance_scores or all(score == 1.0 for score in performance_scores.values()):
                logger.debug("No performance data available, using base allocation")
                return adjusted
            
            # Adjust weights based on performance
            total_performance = sum(performance_scores.values())
            if total_performance > 0:
                for strategy in adjusted.keys():
                    try:
                        score = performance_scores.get(strategy, 1.0)
                        performance_factor = score / (total_performance / len(performance_scores))
                        
                        # Apply learning rate to avoid dramatic changes
                        adjustment = (performance_factor - 1) * self.learning_rate
                        adjusted[strategy] *= (1 + adjustment)
                        adjusted[strategy] = max(0.01, adjusted[strategy])  # Minimum 1%
                    except Exception as e:
                        logger.warning(f"Error adjusting weight for {strategy}: {e}")
                        # Keep original weight
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Error in adjust_for_performance: {e}")
            return base_allocation.copy()
    
    def adjust_for_market_conditions(self, allocation: Dict[str, float], 
                                   market_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust allocation based on current market conditions.
        
        Args:
            allocation: Current allocation weights
            market_conditions: Market metrics (volatility, volume, trend)
            
        Returns:
            Market-adjusted allocation weights
        """
        try:
            adjusted = allocation.copy()
            
            # Extract and validate market conditions
            volatility = market_conditions.get('volatility', 0.02)
            volume_ratio = market_conditions.get('volume_ratio', 1.0)
            trend_strength = market_conditions.get('trend_strength', 0.0)
            
            # Validate bounds
            volatility = max(0.0, min(1.0, volatility))
            volume_ratio = max(0.1, min(10.0, volume_ratio))
            trend_strength = max(-1.0, min(1.0, trend_strength))
            
            # High volatility adjustments
            if volatility > 0.04:
                adjusted['momentum'] = adjusted.get('momentum', 0.25) * 1.2
                adjusted['arbitrage'] = adjusted.get('arbitrage', 0.25) * 1.1
                adjusted['market_making'] = adjusted.get('market_making', 0.25) * 0.8
                
            # Low volatility adjustments
            elif volatility < 0.015:
                adjusted['mean_reversion'] = adjusted.get('mean_reversion', 0.25) * 1.3
                adjusted['market_making'] = adjusted.get('market_making', 0.25) * 1.2
                adjusted['momentum'] = adjusted.get('momentum', 0.25) * 0.7
            
            # High volume benefits all strategies
            if volume_ratio > 1.5:
                for strategy in adjusted.keys():
                    adjusted[strategy] *= 1.1
            
            # Low volume penalizes market making more
            elif volume_ratio < 0.5:
                adjusted['market_making'] = adjusted.get('market_making', 0.25) * 0.6
                adjusted['arbitrage'] = adjusted.get('arbitrage', 0.25) * 0.8
            
            # Strong trend adjustments
            if abs(trend_strength) > 0.02:
                adjusted['momentum'] = adjusted.get('momentum', 0.25) * 1.2
                adjusted['mean_reversion'] = adjusted.get('mean_reversion', 0.25) * 0.8
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Error in adjust_for_market_conditions: {e}")
            return allocation.copy()
    
    def adjust_for_signal_strength(self, allocation: Dict[str, float], 
                                 strategy_signals: Dict[str, List]) -> Dict[str, float]:
        """
        Adjust allocation based on current signal strength.
        
        Args:
            allocation: Current allocation weights
            strategy_signals: Signals from each strategy
            
        Returns:
            Signal-adjusted allocation weights
        """
        try:
            adjusted = allocation.copy()
            
            for strategy, signals in strategy_signals.items():
                if strategy in adjusted:
                    try:
                        # Validate signals
                        if not signals or not isinstance(signals, list):
                            continue
                        
                        # Calculate average signal confidence
                        valid_confidences = []
                        for signal in signals:
                            if hasattr(signal, 'confidence'):
                                confidence = getattr(signal, 'confidence', 0)
                                if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                                    valid_confidences.append(confidence)
                        
                        if valid_confidences:
                            avg_confidence = np.mean(valid_confidences)
                            # Boost allocation for strategies with strong signals
                            confidence_factor = min(avg_confidence / 0.6, 1.5)  # Cap at 1.5x
                            adjusted[strategy] *= confidence_factor
                            
                    except Exception as e:
                        logger.warning(f"Error processing signals for {strategy}: {e}")
                        continue
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Error in adjust_for_signal_strength: {e}")
            return allocation.copy()
    
    def normalize_allocation(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize allocation weights to sum to 1.0.
        
        Args:
            allocation: Unnormalized allocation weights
            
        Returns:
            Normalized allocation weights
        """
        try:
            if not allocation:
                return allocation
            
            total_weight = sum(allocation.values())
            
            if total_weight > 0:
                normalized = {k: v/total_weight for k, v in allocation.items()}
            else:
                # Equal weights if total is zero
                equal_weight = 1.0 / len(allocation)
                normalized = {k: equal_weight for k in allocation}
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing allocation: {e}")
            return allocation
    
    def calculate_strategy_confidence(self, strategy: str, regime_confidence: float,
                                    performance_score: float,
                                    market_volatility: float = 0.02) -> float:
        """
        Calculate confidence score for a strategy.
        
        Args:
            strategy: Strategy name
            regime_confidence: Confidence in regime detection
            performance_score: Recent performance score
            market_volatility: Current market volatility
            
        Returns:
            Confidence score (0-1)
        """
        try:
            base_confidence = 0.5
            
            # Regime alignment
            regime_confidence = max(0.0, min(1.0, regime_confidence))
            base_confidence += regime_confidence * 0.3
            
            # Recent performance
            base_confidence += (performance_score - 1) * 0.2
            
            # Market condition suitability
            volatility = max(0.0, min(1.0, market_volatility))
            
            if strategy == 'momentum' and volatility > 0.03:
                base_confidence += 0.1
            elif strategy == 'mean_reversion' and volatility < 0.02:
                base_confidence += 0.1
            elif strategy == 'arbitrage':
                base_confidence += 0.05  # Always somewhat suitable
            elif strategy == 'market_making' and 0.01 < volatility < 0.03:
                base_confidence += 0.1
            
            return max(0.1, min(0.95, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence for {strategy}: {e}")
            return 0.5  # Default confidence
