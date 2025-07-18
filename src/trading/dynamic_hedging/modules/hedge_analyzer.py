"""
Hedge Analyzer Module

Analyzes portfolio risks and generates hedge recommendations.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from .hedge_types import HedgeRecommendation, HedgePosition

logger = logging.getLogger(__name__)


class HedgeAnalyzer:
    """
    Analyzes portfolio risks and determines appropriate hedging strategies.
    
    Evaluates various risk metrics and generates specific hedge
    recommendations based on configurable thresholds.
    """
    
    def __init__(self):
        """Initialize the hedge analyzer."""
        self.hedge_effectiveness_history = []
        self.recommendation_history = []
    
    def check_beta_hedge(
        self,
        exposure_metrics: Dict[str, Any],
        risk_metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if beta hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            risk_metrics: Current risk metrics
            thresholds: Hedging thresholds
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            portfolio_beta = exposure_metrics.get('portfolio_beta', 0)
            
            if not isinstance(portfolio_beta, (int, float)) or portfolio_beta < 0:
                logger.warning(f"Invalid portfolio beta: {portfolio_beta}")
                return None
            
            threshold = thresholds.get('portfolio_beta_threshold', 1.2)
            
            if portfolio_beta > threshold:
                # Calculate hedge size
                target_beta = 1.0
                excess_beta = portfolio_beta - target_beta
                portfolio_value = exposure_metrics.get('total_value', 0)
                
                if portfolio_value <= 0:
                    logger.warning("Invalid portfolio value for beta hedge calculation")
                    return None
                
                hedge_ratio = excess_beta / portfolio_beta
                hedge_size = portfolio_value * hedge_ratio
                
                # Reasonable bounds check
                if hedge_size <= 0 or hedge_size > portfolio_value:
                    logger.warning(f"Unreasonable hedge size calculated: {hedge_size}")
                    return None
                
                # Get BTC price for hedge sizing
                btc_price = self._get_btc_price(exposure_metrics)
                
                return HedgeRecommendation(
                    hedge_type='beta_hedge',
                    symbol='BTC-PERP',  # Use BTC perpetual as hedge
                    side='sell',
                    size=hedge_size / btc_price,
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'Portfolio beta {portfolio_beta:.2f} exceeds threshold '
                        f'{threshold}'
                    ),
                    urgency='medium',
                    expected_cost=hedge_size * 0.0005,  # 5 bps cost
                    expected_protection=hedge_size * 0.8,  # 80% effectiveness
                    target_metric='portfolio_beta',
                    threshold_breached=portfolio_beta,
                    confidence=0.85
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking beta hedge: {e}")
            return None
    
    def check_correlation_hedge(
        self,
        exposure_metrics: Dict[str, Any],
        market_data: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if correlation hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            market_data: Current market data
            thresholds: Hedging thresholds
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            avg_correlation = exposure_metrics.get('avg_pairwise_correlation', 0)
            threshold = thresholds.get('correlation_threshold', 0.8)
            
            if avg_correlation > threshold:
                portfolio_value = exposure_metrics.get('total_value', 0)
                
                if portfolio_value <= 0:
                    return None
                
                # Hedge 20% of portfolio with low-correlation assets
                hedge_ratio = 0.2
                hedge_size = portfolio_value * hedge_ratio
                
                return HedgeRecommendation(
                    hedge_type='correlation_hedge',
                    symbol='CORRELATION-BASKET',
                    side='buy',  # Buy uncorrelated assets
                    size=hedge_size,
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'Average correlation {avg_correlation:.2f} exceeds '
                        f'threshold {threshold}'
                    ),
                    urgency='low',
                    expected_cost=hedge_size * 0.0008,  # 8 bps cost
                    expected_protection=hedge_size * 0.7,  # 70% effectiveness
                    target_metric='correlation',
                    threshold_breached=avg_correlation,
                    confidence=0.75
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking correlation hedge: {e}")
            return None
    
    def check_volatility_hedge(
        self,
        exposure_metrics: Dict[str, Any],
        risk_metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if volatility hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            risk_metrics: Current risk metrics
            thresholds: Hedging thresholds
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            portfolio_vol = exposure_metrics.get('portfolio_volatility', 0)
            threshold = thresholds.get('volatility_threshold', 0.25)
            
            if portfolio_vol > threshold:
                portfolio_value = exposure_metrics.get('total_value', 0)
                
                if portfolio_value <= 0:
                    return None
                
                # Calculate hedge size based on volatility reduction target
                target_vol = 0.15  # 15% target volatility
                vol_reduction_needed = portfolio_vol - target_vol
                hedge_ratio = vol_reduction_needed / portfolio_vol
                hedge_size = portfolio_value * hedge_ratio * 0.5  # 50% effectiveness
                
                return HedgeRecommendation(
                    hedge_type='volatility_hedge',
                    symbol='VIX-HEDGE',  # Volatility hedge instrument
                    side='buy',
                    size=hedge_size,
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'Portfolio volatility {portfolio_vol:.2%} exceeds '
                        f'threshold {threshold:.2%}'
                    ),
                    urgency='high' if portfolio_vol > threshold * 1.5 else 'medium',
                    expected_cost=hedge_size * 0.0015,  # 15 bps cost
                    expected_protection=hedge_size * 0.6,  # 60% effectiveness
                    target_metric='volatility',
                    threshold_breached=portfolio_vol,
                    confidence=0.8
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking volatility hedge: {e}")
            return None
    
    def check_tail_risk_hedge(
        self,
        exposure_metrics: Dict[str, Any],
        risk_metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if tail risk hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            risk_metrics: Current risk metrics
            thresholds: Hedging thresholds
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            var = exposure_metrics.get('portfolio_var', 0)
            threshold = thresholds.get('var_threshold', 0.05)
            
            if var > threshold:
                portfolio_value = exposure_metrics.get('total_value', 0)
                
                if portfolio_value <= 0:
                    return None
                
                # Hedge against extreme moves (e.g., put options)
                hedge_ratio = 0.1  # Hedge 10% of portfolio
                hedge_size = portfolio_value * hedge_ratio
                
                return HedgeRecommendation(
                    hedge_type='tail_risk_hedge',
                    symbol='PUT-OPTIONS',  # Put options for tail risk
                    side='buy',
                    size=hedge_size,
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'Value at Risk {var:.2%} exceeds threshold {threshold:.2%}'
                    ),
                    urgency='high',
                    expected_cost=hedge_size * 0.002,  # 20 bps cost
                    expected_protection=hedge_size * 2.0,  # 200% on tail event
                    target_metric='var',
                    threshold_breached=var,
                    confidence=0.7,
                    expiry=datetime.now() + timedelta(days=30)  # 30-day protection
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking tail risk hedge: {e}")
            return None
    
    def check_concentration_hedge(
        self,
        exposure_metrics: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> Optional[HedgeRecommendation]:
        """
        Check if concentration hedging is needed.
        
        Args:
            exposure_metrics: Portfolio exposure metrics
            thresholds: Hedging thresholds
            
        Returns:
            HedgeRecommendation if hedging needed, None otherwise
        """
        try:
            max_weight = exposure_metrics.get('max_weight', 0)
            max_sector = exposure_metrics.get('max_sector_concentration', 0)
            
            # Check position concentration
            if max_weight > 0.3:  # 30% in single position
                exposures = exposure_metrics.get('individual_exposures', {})
                
                # Find the largest position
                largest_position = max(
                    exposures.items(),
                    key=lambda x: x[1]['weight']
                )
                
                symbol = largest_position[0]
                weight = largest_position[1]['weight']
                value = largest_position[1]['value']
                
                # Hedge excess concentration
                target_weight = 0.2  # 20% target
                excess_weight = weight - target_weight
                hedge_size = value * (excess_weight / weight)
                
                return HedgeRecommendation(
                    hedge_type='concentration_hedge',
                    symbol=f'{symbol}-HEDGE',
                    side='sell',
                    size=hedge_size / largest_position[1]['price'],
                    hedge_ratio=excess_weight / weight,
                    reason=(
                        f'Position concentration {weight:.1%} in {symbol} '
                        f'exceeds safe level'
                    ),
                    urgency='medium',
                    expected_cost=hedge_size * 0.001,
                    expected_protection=hedge_size * 0.9,
                    target_metric='concentration',
                    threshold_breached=max_weight,
                    confidence=0.9
                )
            
            # Check sector concentration
            elif max_sector > 0.5:  # 50% in single sector
                sector_weights = exposure_metrics.get('sector_weights', {})
                
                # Find most concentrated sector
                largest_sector = max(
                    sector_weights.items(),
                    key=lambda x: x[1]
                )
                
                sector = largest_sector[0]
                weight = largest_sector[1]
                portfolio_value = exposure_metrics.get('total_value', 0)
                
                # Hedge sector concentration
                hedge_ratio = 0.2  # Hedge 20% of sector exposure
                hedge_size = portfolio_value * weight * hedge_ratio
                
                return HedgeRecommendation(
                    hedge_type='concentration_hedge',
                    symbol=f'{sector.upper()}-INVERSE',
                    side='buy',
                    size=hedge_size,
                    hedge_ratio=hedge_ratio,
                    reason=(
                        f'Sector concentration {weight:.1%} in {sector} '
                        f'exceeds safe level'
                    ),
                    urgency='low',
                    expected_cost=hedge_size * 0.0012,
                    expected_protection=hedge_size * 0.75,
                    target_metric='sector_concentration',
                    threshold_breached=max_sector,
                    confidence=0.8
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking concentration hedge: {e}")
            return None
    
    def check_hedge_adjustments(
        self,
        active_hedges: Dict[str, HedgePosition],
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Check if existing hedges need adjustment.
        
        Args:
            active_hedges: Currently active hedge positions
            current_prices: Current market prices
            
        Returns:
            List of adjustment recommendations
        """
        adjustments = []
        
        for hedge_id, position in active_hedges.items():
            try:
                # Check if hedge is still effective
                effectiveness = position.effectiveness_ratio()
                
                if effectiveness < 0.5:  # Less than 50% effective
                    adjustments.append({
                        'hedge_id': hedge_id,
                        'action': 'close',
                        'reason': f'Low effectiveness: {effectiveness:.1%}'
                    })
                
                # Check if hedge is profitable enough to close
                elif position.unrealized_pnl > position.expected_protection * 0.8:
                    adjustments.append({
                        'hedge_id': hedge_id,
                        'action': 'close',
                        'reason': 'Target protection achieved'
                    })
                
                # Check if hedge needs resizing
                elif abs(1 - effectiveness) > 0.3:  # 30% deviation
                    new_size = position.size * (1 / effectiveness)
                    adjustments.append({
                        'hedge_id': hedge_id,
                        'action': 'resize',
                        'new_size': new_size,
                        'reason': f'Effectiveness deviation: {effectiveness:.1%}'
                    })
                
            except Exception as e:
                logger.error(f"Error checking hedge {hedge_id}: {e}")
        
        return adjustments
    
    def analyze_effectiveness(
        self,
        historical_positions: List[HedgePosition]
    ) -> Dict[str, Any]:
        """
        Analyze historical hedge effectiveness.
        
        Args:
            historical_positions: Historical hedge positions
            
        Returns:
            Effectiveness analysis
        """
        if not historical_positions:
            return {'message': 'No historical data'}
        
        try:
            by_type = {}
            
            for position in historical_positions:
                hedge_type = position.hedge_type
                
                if hedge_type not in by_type:
                    by_type[hedge_type] = {
                        'count': 0,
                        'total_cost': 0,
                        'total_protection': 0,
                        'avg_effectiveness': 0,
                        'successful': 0
                    }
                
                stats = by_type[hedge_type]
                stats['count'] += 1
                stats['total_cost'] += position.actual_cost
                stats['total_protection'] += position.actual_protection
                
                effectiveness = position.effectiveness_ratio()
                stats['avg_effectiveness'] += effectiveness
                
                if effectiveness > 0.7:  # 70% effectiveness threshold
                    stats['successful'] += 1
            
            # Calculate averages
            for hedge_type, stats in by_type.items():
                if stats['count'] > 0:
                    stats['avg_effectiveness'] /= stats['count']
                    stats['success_rate'] = stats['successful'] / stats['count']
                    stats['cost_efficiency'] = (
                        stats['total_protection'] / stats['total_cost']
                        if stats['total_cost'] > 0 else 0
                    )
            
            return {
                'by_hedge_type': by_type,
                'total_hedges': len(historical_positions),
                'overall_success_rate': sum(
                    s['successful'] for s in by_type.values()
                ) / len(historical_positions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing effectiveness: {e}")
            return {'error': str(e)}
    
    def analyze_costs(
        self,
        historical_positions: List[HedgePosition]
    ) -> Dict[str, Any]:
        """
        Analyze hedging costs.
        
        Args:
            historical_positions: Historical hedge positions
            
        Returns:
            Cost analysis
        """
        if not historical_positions:
            return {'message': 'No historical data'}
        
        try:
            total_cost = sum(p.actual_cost for p in historical_positions)
            total_protection = sum(p.actual_protection for p in historical_positions)
            total_pnl = sum(p.unrealized_pnl for p in historical_positions)
            
            return {
                'total_cost': total_cost,
                'total_protection': total_protection,
                'total_pnl': total_pnl,
                'net_benefit': total_protection - total_cost,
                'roi': (total_pnl - total_cost) / total_cost if total_cost > 0 else 0,
                'protection_to_cost_ratio': total_protection / total_cost if total_cost > 0 else 0,
                'avg_cost_per_hedge': total_cost / len(historical_positions),
                'avg_protection_per_hedge': total_protection / len(historical_positions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing costs: {e}")
            return {'error': str(e)}
    
    def _get_btc_price(self, exposure_metrics: Dict[str, Any]) -> float:
        """Get BTC price from exposure metrics."""
        exposures = exposure_metrics.get('individual_exposures', {})
        
        # Look for BTC position
        for symbol, exp in exposures.items():
            if 'BTC' in symbol.upper():
                return exp.get('price', 50000)
        
        # Default BTC price
        return 50000
