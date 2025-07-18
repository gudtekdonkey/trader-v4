"""
Performance tracking module for adaptive strategy management.
Handles performance monitoring and scoring for strategies.
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceTracker:
    """Tracks and analyzes strategy performance"""
    
    def __init__(self, performance_window: int = 100, 
                 max_performance_history: int = 500,
                 min_trades_for_performance: int = 10):
        # Performance tracking parameters
        self.performance_window = performance_window
        self.max_performance_history = max_performance_history
        self.min_trades_for_performance = min_trades_for_performance
        
        # Strategy performance data
        self.strategy_performance: Dict[str, Dict[str, Any]] = {
            'momentum': {'returns': [], 'sharpe': 0, 'max_dd': 0},
            'mean_reversion': {'returns': [], 'sharpe': 0, 'max_dd': 0},
            'arbitrage': {'returns': [], 'sharpe': 0, 'max_dd': 0},
            'market_making': {'returns': [], 'sharpe': 0, 'max_dd': 0}
        }
        
    def update_performance(self, strategy: str, trade_return: float) -> None:
        """
        Update strategy performance with new trade return.
        
        Args:
            strategy: Strategy name
            trade_return: Return from the trade
        """
        try:
            # Validate inputs
            if not isinstance(strategy, str) or strategy not in self.strategy_performance:
                logger.warning(f"Invalid strategy name: {strategy}")
                return
            
            if not isinstance(trade_return, (int, float)) or not np.isfinite(trade_return):
                logger.warning(f"Invalid trade return: {trade_return}")
                return
            
            # Update returns
            self.strategy_performance[strategy]['returns'].append(trade_return)
            
            # Keep only recent performance to prevent memory issues
            if len(self.strategy_performance[strategy]['returns']) > self.max_performance_history:
                self.strategy_performance[strategy]['returns'] = \
                    self.strategy_performance[strategy]['returns'][-self.performance_window:]
            
            # Update metrics
            self._update_metrics(strategy)
                    
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def _update_metrics(self, strategy: str) -> None:
        """Update performance metrics for a strategy"""
        try:
            returns = self.strategy_performance[strategy]['returns']
            
            if len(returns) >= 20:
                # Calculate Sharpe ratio
                returns_array = np.array(returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                if std_return > 0:
                    self.strategy_performance[strategy]['sharpe'] = mean_return / std_return
                else:
                    self.strategy_performance[strategy]['sharpe'] = 0
                
                # Calculate max drawdown
                cumulative = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                self.strategy_performance[strategy]['max_dd'] = abs(np.min(drawdowns))
                
        except Exception as e:
            logger.error(f"Error updating metrics for {strategy}: {e}")
    
    def calculate_performance_score(self, strategy: str) -> float:
        """
        Calculate recent performance score for a strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Performance score (1.0 = average)
        """
        try:
            performance = self.strategy_performance.get(strategy, {'returns': []})
            returns = performance.get('returns', [])
            
            # Validate returns data
            if not isinstance(returns, list):
                logger.warning(f"Invalid returns data for {strategy}")
                return 1.0
            
            if len(returns) < self.min_trades_for_performance:
                return 1.0  # Neutral score
            
            # Use recent returns
            recent_returns = returns[-self.performance_window:]
            
            # Filter out invalid returns
            valid_returns = []
            for ret in recent_returns:
                if isinstance(ret, (int, float)) and np.isfinite(ret):
                    valid_returns.append(ret)
            
            if not valid_returns:
                return 1.0
            
            # Calculate metrics
            avg_return = np.mean(valid_returns)
            volatility = np.std(valid_returns)
            
            if volatility > 0:
                sharpe_ratio = avg_return / volatility
                # Convert to performance score (1.0 = average)
                performance_score = 1.0 + (sharpe_ratio - 1.0) * 0.5
            else:
                performance_score = 1.0 if avg_return >= 0 else 0.5
            
            return max(0.1, min(2.0, performance_score))
            
        except Exception as e:
            logger.error(f"Error calculating performance score for {strategy}: {e}")
            return 1.0  # Neutral score
    
    def get_performance_scores(self, strategies: List[str]) -> Dict[str, float]:
        """Get performance scores for multiple strategies"""
        scores = {}
        
        for strategy in strategies:
            try:
                scores[strategy] = self.calculate_performance_score(strategy)
            except Exception as e:
                logger.warning(f"Error getting score for {strategy}: {e}")
                scores[strategy] = 1.0
                
        return scores
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get performance analytics for all strategies"""
        try:
            analytics = {
                'strategy_performance': self.strategy_performance.copy(),
                'performance_window': self.performance_window
            }
            
            # Calculate current strategy rankings
            rankings = {}
            for strategy, perf in self.strategy_performance.items():
                try:
                    returns = perf.get('returns', [])
                    if len(returns) >= self.min_trades_for_performance:
                        recent_returns = returns[-50:] if len(returns) >= 50 else returns
                        
                        # Filter valid returns
                        valid_returns = [r for r in recent_returns 
                                       if isinstance(r, (int, float)) and np.isfinite(r)]
                        
                        if valid_returns:
                            rankings[strategy] = {
                                'avg_return': np.mean(valid_returns),
                                'sharpe_ratio': perf.get('sharpe', 0),
                                'max_drawdown': perf.get('max_dd', 0),
                                'trade_count': len(returns)
                            }
                except Exception as e:
                    logger.error(f"Error calculating ranking for {strategy}: {e}")
                    continue
            
            analytics['current_rankings'] = rankings
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {
                'error': str(e),
                'strategy_performance': {},
                'current_rankings': {}
            }
