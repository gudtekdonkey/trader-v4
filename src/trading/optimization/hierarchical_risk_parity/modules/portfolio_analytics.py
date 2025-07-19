"""
Portfolio analytics module for Hierarchical Risk Parity optimization.
Calculates portfolio metrics, backtesting, and comparisons.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PortfolioAnalytics:
    """Handles portfolio analytics and performance metrics"""
    
    def __init__(self, numerical_tolerance: float = 1e-10):
        self.numerical_tolerance = numerical_tolerance
        
    def calculate_portfolio_metrics(self, returns: pd.DataFrame, 
                                  weights: np.ndarray) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of portfolio weights
            
        Returns:
            Dictionary of portfolio metrics
        """
        try:
            # Validate inputs
            if len(weights) != len(returns.columns):
                logger.error("Weights and returns dimension mismatch")
                return {'error': 'Dimension mismatch'}
            
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Handle empty returns
            if len(portfolio_returns) == 0:
                return {'error': 'No portfolio returns calculated'}
            
            # Basic metrics
            total_return = (1 + portfolio_returns).prod() - 1
            
            # Annualized metrics
            if len(portfolio_returns) > 1:
                volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
                
                # Sharpe ratio
                if volatility > self.numerical_tolerance:
                    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0
            else:
                volatility = 0
                sharpe_ratio = 0
            
            # Diversification metrics
            individual_volatilities = returns.std() * np.sqrt(252)
            weighted_avg_vol = (weights * individual_volatilities).sum()
            
            # Diversification ratio
            if volatility > self.numerical_tolerance:
                diversification_ratio = weighted_avg_vol / volatility
            else:
                diversification_ratio = 1.0
            
            # Risk metrics
            if len(portfolio_returns) > 5:
                var_95 = np.percentile(portfolio_returns, 5)
                cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            else:
                var_95 = 0
                cvar_95 = 0
            
            # Concentration metrics
            concentration = (weights ** 2).sum()  # Herfindahl index
            effective_assets = 1 / concentration if concentration > 0 else len(weights)
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # Sortino ratio (downside deviation)
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'diversification_ratio': diversification_ratio,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_drawdown,
                'concentration': concentration,
                'effective_assets': effective_assets,
                'max_weight': weights.max(),
                'min_weight': weights.min(),
                'avg_weight': weights.mean(),
                'weight_std': weights.std()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                'error': str(e),
                'total_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0
            }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            return drawdowns.min()
        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, 
                                target_return: float = 0) -> float:
        """Calculate Sortino ratio using downside deviation"""
        try:
            excess_returns = returns - target_return
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > self.numerical_tolerance:
                    sortino = returns.mean() / downside_std * np.sqrt(252)
                    return sortino
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error calculating Sortino ratio: {e}")
            return 0
    
    def compare_allocations(self, returns: pd.DataFrame, 
                          hrp_weights: Dict[str, float]) -> pd.DataFrame:
        """
        Compare HRP with other allocation methods.
        
        Args:
            returns: DataFrame of asset returns
            hrp_weights: HRP weight dictionary
            
        Returns:
            DataFrame comparing different allocation methods
        """
        try:
            # Equal weight allocation
            n_assets = len(returns.columns)
            equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
            
            # Inverse volatility allocation
            inv_vol_weights = self._calculate_inverse_volatility_weights(returns)
            
            # Minimum variance weights (simplified)
            min_var_weights = self._calculate_min_variance_weights(returns)
            
            # Risk parity weights
            risk_parity_weights = self._calculate_risk_parity_weights(returns)
            
            # Combine results
            comparison_df = pd.DataFrame({
                'HRP': pd.Series(hrp_weights),
                'Equal_Weight': pd.Series(equal_weights),
                'Inverse_Volatility': pd.Series(inv_vol_weights),
                'Min_Variance': pd.Series(min_var_weights),
                'Risk_Parity': pd.Series(risk_parity_weights)
            }).fillna(0)
            
            # Ensure weights sum to 1
            for col in comparison_df.columns:
                col_sum = comparison_df[col].sum()
                if col_sum > 0:
                    comparison_df[col] = comparison_df[col] / col_sum
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing allocations: {e}")
            return pd.DataFrame()
    
    def _calculate_inverse_volatility_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate inverse volatility weights"""
        try:
            volatilities = returns.std()
            # Handle zero volatility
            volatilities = volatilities.replace(0, volatilities.mean())
            if volatilities.sum() > 0:
                inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
                return inv_vol_weights.to_dict()
            else:
                n_assets = len(returns.columns)
                return {asset: 1.0/n_assets for asset in returns.columns}
        except Exception as e:
            logger.warning(f"Error calculating inverse volatility weights: {e}")
            n_assets = len(returns.columns)
            return {asset: 1.0/n_assets for asset in returns.columns}
    
    def _calculate_min_variance_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate minimum variance weights (simplified version)"""
        try:
            # This is a simplified version - just uses inverse of variance
            variances = returns.var()
            variances = variances.replace(0, variances.mean())
            
            if variances.sum() > 0:
                weights = (1 / variances) / (1 / variances).sum()
                return weights.to_dict()
            else:
                n_assets = len(returns.columns)
                return {asset: 1.0/n_assets for asset in returns.columns}
                
        except Exception as e:
            logger.warning(f"Error calculating min variance weights: {e}")
            n_assets = len(returns.columns)
            return {asset: 1.0/n_assets for asset in returns.columns}
    
    def _calculate_risk_parity_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk parity weights"""
        try:
            # Simple risk parity: equal risk contribution
            volatilities = returns.std()
            volatilities = volatilities.replace(0, volatilities.mean())
            
            # Initial guess: inverse volatility
            weights = 1 / volatilities
            weights = weights / weights.sum()
            
            # Iterative adjustment (simplified)
            for _ in range(10):
                # Calculate risk contributions
                portfolio_vol = np.sqrt((weights * returns).sum(axis=1).var())
                marginal_contrib = (returns * weights).cov().dot(weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                
                # Adjust weights
                target_contrib = 1.0 / len(weights)
                weights = weights * target_contrib / risk_contrib
                weights = weights / weights.sum()
            
            return weights.to_dict()
            
        except Exception as e:
            logger.warning(f"Error calculating risk parity weights: {e}")
            n_assets = len(returns.columns)
            return {asset: 1.0/n_assets for asset in returns.columns}
    
    def backtest_strategy(self, returns: pd.DataFrame, 
                        weight_function,
                        rebalance_freq: str = 'M',
                        lookback_days: int = 60) -> pd.Series:
        """
        Backtest a strategy with periodic rebalancing.
        
        Args:
            returns: DataFrame of asset returns
            weight_function: Function that calculates weights given returns
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            lookback_days: Days of history for weight calculation
            
        Returns:
            Series of strategy returns
        """
        try:
            # Validate inputs
            valid_frequencies = ['D', 'W', 'M', 'Q', 'Y']
            if rebalance_freq not in valid_frequencies:
                logger.warning(f"Invalid rebalance frequency {rebalance_freq}, using 'M'")
                rebalance_freq = 'M'
            
            # Get rebalancing dates
            rebalance_dates = self._get_rebalance_dates(returns, rebalance_freq)
            
            portfolio_returns = []
            current_weights = None
            
            for date in returns.index:
                try:
                    if date in rebalance_dates or current_weights is None:
                        # Rebalance portfolio
                        lookback_returns = returns.loc[:date].tail(lookback_days)
                        
                        if len(lookback_returns) >= 30:  # Minimum data requirement
                            weights_dict = weight_function(lookback_returns)
                            
                            if weights_dict:
                                current_weights = np.array([
                                    weights_dict.get(asset, 0) 
                                    for asset in returns.columns
                                ])
                            else:
                                # Use equal weights if optimization fails
                                current_weights = np.ones(len(returns.columns)) / len(returns.columns)
                        else:
                            # Equal weights if insufficient data
                            current_weights = np.ones(len(returns.columns)) / len(returns.columns)
                    
                    # Calculate portfolio return for the day
                    if current_weights is not None:
                        daily_return = (returns.loc[date] * current_weights).sum()
                        portfolio_returns.append(daily_return)
                    else:
                        portfolio_returns.append(0)
                        
                except Exception as e:
                    logger.warning(f"Error calculating return for {date}: {e}")
                    portfolio_returns.append(0)
            
            return pd.Series(portfolio_returns, index=returns.index, name='Strategy_Returns')
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return pd.Series(0, index=returns.index, name='Strategy_Returns')
    
    def _get_rebalance_dates(self, returns: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
        """Get rebalancing dates based on frequency"""
        try:
            if freq == 'M':
                return returns.resample('M').last().index
            elif freq == 'Q':
                return returns.resample('Q').last().index
            elif freq == 'W':
                return returns.resample('W').last().index
            elif freq == 'Y':
                return returns.resample('Y').last().index
            else:  # Daily
                return returns.index
        except Exception as e:
            logger.error(f"Error getting rebalance dates: {e}")
            return returns.index
