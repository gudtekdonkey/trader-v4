"""
Dynamic Hedging System
Implements portfolio hedging strategies to manage downside risk
"""

"""
Dynamic Hedging System Module

Implements portfolio hedging strategies to manage downside risk.
This module provides automated hedging recommendations and execution
based on portfolio exposure, market conditions, and risk metrics.

Classes:
    HedgeRecommendation: Data class for hedge recommendations
    DynamicHedgingSystem: Main hedging system implementation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HedgeRecommendation:
    """
    Hedge recommendation structure.
    
    Attributes:
        hedge_type: Type of hedge (beta_hedge, volatility_hedge, etc.)
        symbol: Instrument symbol for hedging
        side: Trade side (buy/sell)
        size: Position size for hedge
        hedge_ratio: Ratio of hedge to portfolio
        reason: Explanation for hedge recommendation
        urgency: Urgency level (low/medium/high)
        expected_cost: Expected cost of hedge in base currency
        expected_protection: Expected protection value
    """
    hedge_type: str
    symbol: str
    side: str
    size: float
    hedge_ratio: float
    reason: str
    urgency: str
    expected_cost: float
    expected_protection: float


class DynamicHedgingSystem:
    """
    Dynamic portfolio hedging system.
    
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
        """Initialize the Dynamic Hedging System."""
        self.hedge_positions: Dict[str, Dict] = {}
        self.hedge_history: List[Dict] = []
        
        # Hedging parameters
        self.hedge_thresholds: Dict[str, float] = {
            'portfolio_beta_threshold': 1.2,  # Hedge when portfolio beta > 1.2
            'correlation_threshold': 0.8,     # Hedge when correlation > 0.8
            'volatility_threshold': 0.25,     # Hedge when volatility > 25%
            'drawdown_threshold': 0.08,       # Hedge when drawdown > 8%
            'var_threshold': 0.05             # Hedge when VaR > 5%
        }
        
        # Hedge instruments (simplified)
        self.hedge_instruments: Dict[str, Dict[str, any]] = {
            'short_futures': {
                'symbols': ['BTC-PERP', 'ETH-PERP'],
                'cost_bps': 5,  # 5 basis points cost
                'effectiveness': 0.9
            },
            'put_options': {
                'symbols': ['BTC-PUT', 'ETH-PUT'],
                'cost_bps': 15,  # 15 basis points cost
                'effectiveness': 0.8
            },
            'correlation_hedge': {
                'symbols': ['CORRELATION-BASKET'],
                'cost_bps': 8,
                'effectiveness': 0.7
            }
        }
    
    def analyze_hedge_needs(
        self, 
        portfolio_positions: Dict[str, Dict],
        market_data: Dict[str, any],
        risk_metrics: Dict[str, float]
    ) -> List[HedgeRecommendation]:
        """
        Analyze portfolio and recommend hedging strategies.
        
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
            recommendations = []
            
            # Calculate portfolio exposure metrics
            exposure_metrics = self._calculate_exposure_metrics(
                portfolio_positions, 
                market_data
            )
            
            # Check for beta hedging needs
            beta_hedge = self._check_beta_hedge(exposure_metrics, risk_metrics)
            if beta_hedge:
                recommendations.append(beta_hedge)
            
            # Check for correlation hedging needs
            correlation_hedge = self._check_correlation_hedge(
                exposure_metrics, 
                market_data
            )
            if correlation_hedge:
                recommendations.append(correlation_hedge)
            
            # Check for volatility hedging needs
            volatility_hedge = self._check_volatility_hedge(
                exposure_metrics, 
                risk_metrics
            )
            if volatility_hedge:
                recommendations.append(volatility_hedge)
            
            # Check for tail risk hedging
            tail_risk_hedge = self._check_tail_risk_hedge(
                exposure_metrics, 
                risk_metrics
            )
            if tail_risk_hedge:
                recommendations.append(tail_risk_hedge)
            
            # Check for sector/theme concentration hedging
            concentration_hedge = self._check_concentration_hedge(exposure_metrics)
            if concentration_hedge:
                recommendations.append(concentration_hedge)
            
            logger.info(f"Generated {len(recommendations)} hedge recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing hedge needs: {e}")
            return []
    
    def _calculate_exposure_metrics(
        self, 
        positions: Dict[str, Dict], 
        market_data: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Calculate portfolio exposure metrics.
        
        Args:
            positions: Portfolio positions
            market_data: Current market data
            
        Returns:
            Dictionary containing exposure metrics
        """
        try:
            total_value = sum(
                pos['size'] * pos.get('current_price', pos['entry_price'])
                for pos in positions.values()
            )
            
            # Calculate individual exposures
            exposures = {}
            for symbol, position in positions.items():
                current_price = position.get('current_price', position['entry_price'])
                value = position['size'] * current_price
                exposures[symbol] = {
                    'weight': value / total_value if total_value > 0 else 0,
                    'value': value,
                    'beta': self._estimate_beta(symbol, market_data),
                    'volatility': self._estimate_volatility(symbol, market_data)
                }
            
            # Calculate portfolio-level metrics
            portfolio_beta = sum(
                exp['weight'] * exp['beta'] 
                for exp in exposures.values()
            )
            
            portfolio_volatility = np.sqrt(sum(
                (exp['weight'] * exp['volatility']) ** 2 
                for exp in exposures.values()
            ))
            
            # Concentration metrics
            weights = [exp['weight'] for exp in exposures.values()]
            max_weight = max(weights) if weights else 0
            herfindahl_index = sum(w**2 for w in weights)
            
            return {
                'total_value': total_value,
                'individual_exposures': exposures,
                'portfolio_beta': portfolio_beta,
                'portfolio_volatility': portfolio_volatility,
                'max_weight': max_weight,
                'concentration_index': herfindahl_index,
                'effective_positions': 1 / herfindahl_index if herfindahl_index > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating exposure metrics: {e}")
            return {}
    
    def _estimate_beta(self, symbol: str, market_data: Dict[str, any]) -> float:
        """
        Estimate asset beta relative to market.
        
        Args:
            symbol: Asset symbol
            market_data: Market data
            
        Returns:
            Estimated beta coefficient
        """
        try:
            # Simplified beta estimation
            # In practice, would calculate against market benchmark
            
            if 'BTC' in symbol:
                return 1.0  # Bitcoin as market proxy
            elif 'ETH' in symbol:
                return 1.2  # Ethereum typically higher beta
            elif 'SOL' in symbol or 'AVAX' in symbol:
                return 1.5  # Alt coins higher beta
            else:
                return 1.1  # Default
                
        except Exception as e:
            logger.error(f"Error estimating beta for {symbol}: {e}")
            return 1.0
    
    def _estimate_volatility(self, symbol: str, market_data: Dict[str, any]) -> float:
        """
        Estimate asset volatility.
        
        Args:
            symbol: Asset symbol
            market_data: Market data
            
        Returns:
            Annualized volatility estimate
        """
        try:
            # Simplified volatility estimation
            # In practice, would calculate from historical returns
            
            if 'BTC' in symbol:
                return 0.6  # 60% annual volatility
            elif 'ETH' in symbol:
                return 0.8  # 80% annual volatility
            else:
                return 1.0  # 100% for alts
                
        except Exception as e:
            logger.error(f"Error estimating volatility for {symbol}: {e}")
            return 0.6
    
    def _check_beta_hedge(
        self, 
        exposure_metrics: Dict[str, any], 
        risk_metrics: Dict[str, float]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if beta hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            risk_metrics: Current risk metrics
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            portfolio_beta = exposure_metrics.get('portfolio_beta', 0)
            
            if portfolio_beta > self.hedge_thresholds['portfolio_beta_threshold']:
                # Calculate hedge size to reduce beta to 1.0
                target_beta = 1.0
                excess_beta = portfolio_beta - target_beta
                portfolio_value = exposure_metrics.get('total_value', 0)
                
                hedge_ratio = excess_beta / portfolio_beta
                hedge_size = portfolio_value * hedge_ratio
                
                return HedgeRecommendation(
                    hedge_type='beta_hedge',
                    symbol='BTC-PERP',  # Use BTC perpetual as hedge
                    side='sell',
                    size=hedge_size / 50000,  # Assume $50k BTC price
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'Portfolio beta {portfolio_beta:.2f} exceeds threshold '
                        f'{self.hedge_thresholds["portfolio_beta_threshold"]}'
                    ),
                    urgency='medium',
                    expected_cost=hedge_size * 0.0005,  # 5 bps cost
                    expected_protection=hedge_size * 0.8  # 80% effectiveness
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking beta hedge: {e}")
            return None
    
    def _check_correlation_hedge(
        self, 
        exposure_metrics: Dict[str, any], 
        market_data: Dict[str, any]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if correlation hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            market_data: Current market data
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            # Calculate average correlation between positions
            exposures = exposure_metrics.get('individual_exposures', {})
            
            if len(exposures) < 2:
                return None
            
            # Simplified correlation calculation
            # In practice, would use historical correlation matrix
            symbols = list(exposures.keys())
            correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    # Estimate correlation based on asset types
                    if (('BTC' in symbol1 and 'ETH' in symbol2) or 
                        ('ETH' in symbol1 and 'BTC' in symbol2)):
                        corr = 0.7
                    elif 'BTC' in symbol1 or 'BTC' in symbol2:
                        corr = 0.6
                    else:
                        corr = 0.8  # High correlation between alts
                    
                    correlations.append(corr)
            
            avg_correlation = np.mean(correlations) if correlations else 0
            
            if avg_correlation > self.hedge_thresholds['correlation_threshold']:
                portfolio_value = exposure_metrics.get('total_value', 0)
                hedge_ratio = 0.2  # Hedge 20% of portfolio
                hedge_size = portfolio_value * hedge_ratio
                
                return HedgeRecommendation(
                    hedge_type='correlation_hedge',
                    symbol='CORRELATION-BASKET',
                    side='sell',
                    size=hedge_size,
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'Average correlation {avg_correlation:.2f} exceeds threshold '
                        f'{self.hedge_thresholds["correlation_threshold"]}'
                    ),
                    urgency='low',
                    expected_cost=hedge_size * 0.0008,  # 8 bps cost
                    expected_protection=hedge_size * 0.7  # 70% effectiveness
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking correlation hedge: {e}")
            return None
    
    def _check_volatility_hedge(
        self, 
        exposure_metrics: Dict[str, any], 
        risk_metrics: Dict[str, float]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if volatility hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            risk_metrics: Current risk metrics
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            portfolio_volatility = exposure_metrics.get('portfolio_volatility', 0)
            
            if portfolio_volatility > self.hedge_thresholds['volatility_threshold']:
                portfolio_value = exposure_metrics.get('total_value', 0)
                
                # Calculate hedge to reduce volatility
                target_volatility = self.hedge_thresholds['volatility_threshold']
                vol_reduction_needed = portfolio_volatility - target_volatility
                hedge_ratio = vol_reduction_needed / portfolio_volatility
                hedge_size = portfolio_value * hedge_ratio
                
                return HedgeRecommendation(
                    hedge_type='volatility_hedge',
                    symbol='BTC-PUT',  # Use put options for vol hedge
                    side='buy',
                    size=hedge_size / 50000,  # Convert to BTC terms
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'Portfolio volatility {portfolio_volatility:.2%} exceeds threshold '
                        f'{self.hedge_thresholds["volatility_threshold"]:.2%}'
                    ),
                    urgency='medium',
                    expected_cost=hedge_size * 0.0015,  # 15 bps cost for options
                    expected_protection=hedge_size * 0.8
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking volatility hedge: {e}")
            return None
    
    def _check_tail_risk_hedge(
        self, 
        exposure_metrics: Dict[str, any], 
        risk_metrics: Dict[str, float]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if tail risk hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            risk_metrics: Current risk metrics
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            # Check VaR and current drawdown
            var_95 = risk_metrics.get('var_95', 0)
            current_drawdown = risk_metrics.get('current_drawdown', 0)
            
            if (abs(var_95) > self.hedge_thresholds['var_threshold'] or 
                current_drawdown > self.hedge_thresholds['drawdown_threshold']):
                
                portfolio_value = exposure_metrics.get('total_value', 0)
                hedge_ratio = 0.1  # Tail risk hedge 10% of portfolio
                hedge_size = portfolio_value * hedge_ratio
                
                return HedgeRecommendation(
                    hedge_type='tail_risk_hedge',
                    symbol='BTC-PUT',
                    side='buy',
                    size=hedge_size / 50000,
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'VaR {abs(var_95):.2%} or drawdown {current_drawdown:.2%} '
                        f'exceeds thresholds'
                    ),
                    urgency='high',
                    expected_cost=hedge_size * 0.002,  # 20 bps for tail protection
                    expected_protection=hedge_size * 0.9
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking tail risk hedge: {e}")
            return None
    
    def _check_concentration_hedge(
        self, 
        exposure_metrics: Dict[str, any]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if concentration hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            max_weight = exposure_metrics.get('max_weight', 0)
            
            if max_weight > 0.4:  # If single position > 40%
                # Find the largest position
                exposures = exposure_metrics.get('individual_exposures', {})
                largest_position = max(
                    exposures.items(), 
                    key=lambda x: x[1]['weight']
                )
                symbol, exposure = largest_position
                
                portfolio_value = exposure_metrics.get('total_value', 0)
                hedge_ratio = 0.5  # Hedge 50% of the concentrated position
                hedge_size = exposure['value'] * hedge_ratio
                
                return HedgeRecommendation(
                    hedge_type='concentration_hedge',
                    symbol=f'{symbol}-PERP',
                    side='sell',
                    size=hedge_size / exposure.get('current_price', 50000),
                    hedge_ratio=hedge_ratio,
                    reason=f'Position {symbol} represents {max_weight:.1%} of portfolio',
                    urgency='medium',
                    expected_cost=hedge_size * 0.0005,
                    expected_protection=hedge_size * 0.9
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking concentration hedge: {e}")
            return None
    
    async def execute_hedge(
        self, 
        recommendation: HedgeRecommendation, 
        order_executor
    ) -> Dict[str, any]:
        """
        Execute a hedge recommendation.
        
        Args:
            recommendation: Hedge recommendation to execute
            order_executor: Order execution interface
            
        Returns:
            Execution result dictionary
        """
        try:
            logger.info(
                f"Executing {recommendation.hedge_type} hedge: "
                f"{recommendation.symbol} {recommendation.side} {recommendation.size:.4f}"
            )
            
            # Execute hedge order
            order_result = await order_executor.place_order(
                symbol=recommendation.symbol,
                side=recommendation.side,
                size=recommendation.size,
                order_type='market',
                metadata={
                    'hedge_type': recommendation.hedge_type,
                    'hedge_ratio': recommendation.hedge_ratio,
                    'reason': recommendation.reason
                }
            )
            
            if order_result['status'] in ['filled', 'partial']:
                # Record hedge position
                hedge_id = f"hedge_{recommendation.hedge_type}_{int(datetime.now().timestamp())}"
                
                hedge_position = {
                    'hedge_id': hedge_id,
                    'hedge_type': recommendation.hedge_type,
                    'symbol': recommendation.symbol,
                    'side': recommendation.side,
                    'size': order_result.get('filled_size', recommendation.size),
                    'entry_price': order_result.get('fill_price', 0),
                    'hedge_ratio': recommendation.hedge_ratio,
                    'timestamp': datetime.now(),
                    'status': 'active',
                    'reason': recommendation.reason,
                    'expected_cost': recommendation.expected_cost,
                    'expected_protection': recommendation.expected_protection
                }
                
                self.hedge_positions[hedge_id] = hedge_position
                self.hedge_history.append(hedge_position.copy())
                
                logger.info(f"Hedge executed successfully: {hedge_id}")
                
                return {
                    'status': 'success',
                    'hedge_id': hedge_id,
                    'hedge_position': hedge_position
                }
            else:
                return {
                    'status': 'failed',
                    'reason': 'Order not filled',
                    'order_result': order_result
                }
                
        except Exception as e:
            logger.error(f"Error executing hedge: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_hedge_positions(self, current_prices: Dict[str, float]) -> None:
        """
        Update hedge position values and P&L.
        
        Args:
            current_prices: Current market prices
        """
        try:
            for hedge_id, position in self.hedge_positions.items():
                if position['status'] != 'active':
                    continue
                
                symbol = position['symbol']
                current_price = current_prices.get(symbol, position['entry_price'])
                
                # Calculate P&L
                if position['side'] == 'buy':
                    pnl = (current_price - position['entry_price']) * position['size']
                else:  # sell
                    pnl = (position['entry_price'] - current_price) * position['size']
                
                position['current_price'] = current_price
                position['unrealized_pnl'] = pnl
                position['last_updated'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating hedge positions: {e}")
    
    def close_hedge(self, hedge_id: str, order_executor) -> Dict[str, any]:
        """
        Close a hedge position.
        
        Args:
            hedge_id: ID of hedge to close
            order_executor: Order execution interface
            
        Returns:
            Close result dictionary
        """
        try:
            if hedge_id not in self.hedge_positions:
                return {'status': 'error', 'message': 'Hedge not found'}
            
            position = self.hedge_positions[hedge_id]
            
            if position['status'] != 'active':
                return {'status': 'error', 'message': 'Hedge not active'}
            
            # Close hedge by taking opposite position
            close_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            # This would execute the closing order
            # For now, just mark as closed
            position['status'] = 'closed'
            position['close_timestamp'] = datetime.now()
            
            logger.info(f"Hedge {hedge_id} closed")
            
            return {'status': 'success', 'hedge_id': hedge_id}
            
        except Exception as e:
            logger.error(f"Error closing hedge {hedge_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_hedge_effectiveness_report(self) -> Dict[str, any]:
        """
        Generate hedge effectiveness report.
        
        Returns:
            Dictionary containing hedge performance metrics
        """
        try:
            if not self.hedge_history:
                return {'message': 'No hedge history available'}
            
            # Analyze hedge performance
            total_hedges = len(self.hedge_history)
            active_hedges = len([
                h for h in self.hedge_positions.values() 
                if h['status'] == 'active'
            ])
            
            # Calculate costs and protection
            total_cost = sum(h.get('expected_cost', 0) for h in self.hedge_history)
            total_protection = sum(
                h.get('expected_protection', 0) 
                for h in self.hedge_history
            )
            
            # Hedge type breakdown
            hedge_types = {}
            for hedge in self.hedge_history:
                hedge_type = hedge['hedge_type']
                if hedge_type not in hedge_types:
                    hedge_types[hedge_type] = 0
                hedge_types[hedge_type] += 1
            
            return {
                'total_hedges': total_hedges,
                'active_hedges': active_hedges,
                'total_cost': total_cost,
                'total_protection': total_protection,
                'cost_efficiency': (
                    total_protection / total_cost if total_cost > 0 else 0
                ),
                'hedge_type_breakdown': hedge_types,
                'average_hedge_ratio': np.mean([
                    h.get('hedge_ratio', 0) for h in self.hedge_history
                ])
            }
            
        except Exception as e:
            logger.error(f"Error generating hedge effectiveness report: {e}")
            return {'error': str(e)}
    
    def update_hedge_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update hedging thresholds.
        
        Args:
            new_thresholds: Dictionary of new threshold values
        """
        try:
            self.hedge_thresholds.update(new_thresholds)
            logger.info(f"Hedge thresholds updated: {new_thresholds}")
        except Exception as e:
            logger.error(f"Error updating hedge thresholds: {e}")


# Example usage
if __name__ == '__main__':
    # Test the hedging system
    hedging_system = DynamicHedgingSystem()
    
    # Mock portfolio positions
    positions = {
        'BTC-USD': {
            'size': 2.0,
            'entry_price': 45000,
            'current_price': 47000,
            'side': 'long'
        },
        'ETH-USD': {
            'size': 10.0,
            'entry_price': 3200,
            'current_price': 3300,
            'side': 'long'
        }
    }
    
    # Mock risk metrics
    risk_metrics = {
        'var_95': 0.06,
        'current_drawdown': 0.12
    }
    
    # Analyze hedge needs
    recommendations = hedging_system.analyze_hedge_needs(
        positions, {}, risk_metrics
    )
    
    print(f"Generated {len(recommendations)} hedge recommendations:")
    for rec in recommendations:
        print(
            f"- {rec.hedge_type}: {rec.symbol} {rec.side} {rec.size:.4f} "
            f"({rec.urgency} urgency)"
        )
        print(f"  Reason: {rec.reason}")
        print(
            f"  Cost: ${rec.expected_cost:.2f}, "
            f"Protection: ${rec.expected_protection:.2f}"
        )