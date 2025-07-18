"""
Hedge Instruments Module

Manages available hedging instruments and their characteristics.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .hedge_types import HedgeInstrument

logger = logging.getLogger(__name__)


class HedgeInstrumentManager:
    """
    Manages available hedging instruments and their specifications.
    
    Provides information about hedging instruments, tracks their
    usage, and recommends appropriate instruments for different
    hedge types.
    """
    
    def __init__(self):
        """Initialize the hedge instrument manager."""
        self.instruments = self._initialize_instruments()
        self.usage_stats = {}
        self.instrument_performance = {}
    
    def _initialize_instruments(self) -> Dict[str, HedgeInstrument]:
        """Initialize available hedging instruments."""
        instruments = {}
        
        # Perpetual futures for directional hedging
        instruments['BTC-PERP'] = HedgeInstrument(
            symbol='BTC-PERP',
            instrument_type='perpetual_future',
            underlying='BTC',
            cost_bps=5,  # 5 basis points
            effectiveness=0.9,
            min_size=0.001,
            max_size=100,
            liquidity_score=1.0
        )
        
        instruments['ETH-PERP'] = HedgeInstrument(
            symbol='ETH-PERP',
            instrument_type='perpetual_future',
            underlying='ETH',
            cost_bps=5,
            effectiveness=0.85,
            min_size=0.01,
            max_size=1000,
            liquidity_score=0.95
        )
        
        # Options for tail risk
        instruments['BTC-PUT-1M'] = HedgeInstrument(
            symbol='BTC-PUT-1M',
            instrument_type='put_option',
            underlying='BTC',
            cost_bps=15,
            effectiveness=0.8,
            min_size=0.1,
            max_size=10,
            expiry=datetime.now() + timedelta(days=30),
            delta=-0.3,
            gamma=0.02,
            vega=0.1,
            liquidity_score=0.7
        )
        
        instruments['ETH-PUT-1M'] = HedgeInstrument(
            symbol='ETH-PUT-1M',
            instrument_type='put_option',
            underlying='ETH',
            cost_bps=18,
            effectiveness=0.75,
            min_size=1,
            max_size=100,
            expiry=datetime.now() + timedelta(days=30),
            delta=-0.3,
            gamma=0.02,
            vega=0.12,
            liquidity_score=0.65
        )
        
        # Volatility products
        instruments['VIX-HEDGE'] = HedgeInstrument(
            symbol='VIX-HEDGE',
            instrument_type='volatility_future',
            underlying='CRYPTO_VIX',
            cost_bps=20,
            effectiveness=0.6,
            min_size=100,
            max_size=10000,
            liquidity_score=0.5
        )
        
        # Correlation hedges
        instruments['CORRELATION-BASKET'] = HedgeInstrument(
            symbol='CORRELATION-BASKET',
            instrument_type='basket',
            underlying='MULTI',
            cost_bps=8,
            effectiveness=0.7,
            min_size=1000,
            max_size=100000,
            liquidity_score=0.8
        )
        
        # Inverse products for specific sectors
        instruments['DEFI-INVERSE'] = HedgeInstrument(
            symbol='DEFI-INVERSE',
            instrument_type='inverse_etf',
            underlying='DEFI_INDEX',
            cost_bps=12,
            effectiveness=0.75,
            min_size=100,
            max_size=50000,
            liquidity_score=0.6
        )
        
        instruments['LAYER1-INVERSE'] = HedgeInstrument(
            symbol='LAYER1-INVERSE',
            instrument_type='inverse_etf',
            underlying='L1_INDEX',
            cost_bps=12,
            effectiveness=0.75,
            min_size=100,
            max_size=50000,
            liquidity_score=0.6
        )
        
        return instruments
    
    def get_instrument(self, symbol: str) -> Optional[HedgeInstrument]:
        """
        Get a specific hedging instrument.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            HedgeInstrument or None if not found
        """
        return self.instruments.get(symbol)
    
    def get_instruments_for_hedge_type(
        self,
        hedge_type: str
    ) -> List[HedgeInstrument]:
        """
        Get suitable instruments for a specific hedge type.
        
        Args:
            hedge_type: Type of hedge needed
            
        Returns:
            List of suitable instruments
        """
        suitable_instruments = []
        
        if hedge_type == 'beta_hedge':
            # Use perpetual futures for beta hedging
            suitable_instruments.extend([
                self.instruments['BTC-PERP'],
                self.instruments['ETH-PERP']
            ])
        
        elif hedge_type == 'tail_risk_hedge':
            # Use put options for tail risk
            suitable_instruments.extend([
                inst for inst in self.instruments.values()
                if inst.instrument_type == 'put_option' and inst.is_available()
            ])
        
        elif hedge_type == 'volatility_hedge':
            # Use volatility products
            if 'VIX-HEDGE' in self.instruments:
                suitable_instruments.append(self.instruments['VIX-HEDGE'])
        
        elif hedge_type == 'correlation_hedge':
            # Use correlation basket or inverse products
            if 'CORRELATION-BASKET' in self.instruments:
                suitable_instruments.append(self.instruments['CORRELATION-BASKET'])
        
        elif hedge_type == 'concentration_hedge':
            # Use inverse ETFs or specific hedges
            suitable_instruments.extend([
                inst for inst in self.instruments.values()
                if 'INVERSE' in inst.symbol
            ])
        
        # Filter by availability and liquidity
        return [
            inst for inst in suitable_instruments
            if inst and inst.is_available() and inst.liquidity_score > 0.4
        ]
    
    def recommend_instrument(
        self,
        hedge_type: str,
        notional_amount: float,
        urgency: str = 'medium'
    ) -> Optional[HedgeInstrument]:
        """
        Recommend the best instrument for a hedge.
        
        Args:
            hedge_type: Type of hedge needed
            notional_amount: Notional amount to hedge
            urgency: Urgency level
            
        Returns:
            Recommended instrument or None
        """
        suitable = self.get_instruments_for_hedge_type(hedge_type)
        
        if not suitable:
            logger.warning(f"No suitable instruments for {hedge_type}")
            return None
        
        # Score instruments based on criteria
        scored_instruments = []
        
        for instrument in suitable:
            score = self._score_instrument(
                instrument, 
                notional_amount, 
                urgency
            )
            scored_instruments.append((score, instrument))
        
        # Sort by score (highest first)
        scored_instruments.sort(key=lambda x: x[0], reverse=True)
        
        if scored_instruments:
            best_instrument = scored_instruments[0][1]
            logger.info(
                f"Recommended {best_instrument.symbol} for {hedge_type} hedge"
            )
            return best_instrument
        
        return None
    
    def _score_instrument(
        self,
        instrument: HedgeInstrument,
        notional_amount: float,
        urgency: str
    ) -> float:
        """Score an instrument based on various criteria."""
        score = 0
        
        # Effectiveness score (40%)
        score += instrument.effectiveness * 40
        
        # Cost efficiency score (30%)
        cost_score = 30 * (1 - min(instrument.cost_bps / 50, 1))  # Lower cost is better
        score += cost_score
        
        # Liquidity score (20%)
        score += instrument.liquidity_score * 20
        
        # Size appropriateness (10%)
        if instrument.min_size <= notional_amount <= instrument.max_size:
            score += 10
        elif notional_amount < instrument.min_size:
            score += 5 * (notional_amount / instrument.min_size)
        else:
            score += 5 * (instrument.max_size / notional_amount)
        
        # Urgency adjustment
        if urgency == 'high' and instrument.liquidity_score > 0.8:
            score *= 1.2  # Prefer liquid instruments for urgent hedges
        elif urgency == 'low' and instrument.cost_bps < 10:
            score *= 1.1  # Prefer cost-effective for non-urgent
        
        return score
    
    def record_usage(
        self,
        symbol: str,
        notional_amount: float,
        effectiveness: float
    ) -> None:
        """
        Record instrument usage for analytics.
        
        Args:
            symbol: Instrument symbol
            notional_amount: Notional amount hedged
            effectiveness: Actual effectiveness achieved
        """
        if symbol not in self.usage_stats:
            self.usage_stats[symbol] = {
                'count': 0,
                'total_notional': 0,
                'avg_effectiveness': 0
            }
        
        stats = self.usage_stats[symbol]
        stats['count'] += 1
        stats['total_notional'] += notional_amount
        
        # Update average effectiveness
        prev_avg = stats['avg_effectiveness']
        stats['avg_effectiveness'] = (
            (prev_avg * (stats['count'] - 1) + effectiveness) / 
            stats['count']
        )
    
    def get_available_instruments(self) -> List[Dict[str, Any]]:
        """Get all available instruments with their specifications."""
        available = []
        
        for symbol, instrument in self.instruments.items():
            if instrument.is_available():
                available.append({
                    'symbol': symbol,
                    'type': instrument.instrument_type,
                    'underlying': instrument.underlying,
                    'cost_bps': instrument.cost_bps,
                    'effectiveness': instrument.effectiveness,
                    'liquidity': instrument.liquidity_score,
                    'min_size': instrument.min_size,
                    'max_size': instrument.max_size,
                    'expiry': instrument.expiry.isoformat() if instrument.expiry else None
                })
        
        return available
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all instruments."""
        return {
            'instrument_usage': self.usage_stats,
            'most_used': max(
                self.usage_stats.items(),
                key=lambda x: x[1]['count']
            )[0] if self.usage_stats else None,
            'most_effective': max(
                self.usage_stats.items(),
                key=lambda x: x[1]['avg_effectiveness']
            )[0] if self.usage_stats else None,
            'total_instruments': len(self.instruments),
            'available_instruments': len([
                i for i in self.instruments.values() 
                if i.is_available()
            ])
        }
    
    def update_instrument_specifications(
        self,
        symbol: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update instrument specifications.
        
        Args:
            symbol: Instrument symbol
            updates: Dictionary of updates
            
        Returns:
            True if successful
        """
        try:
            if symbol not in self.instruments:
                logger.error(f"Instrument {symbol} not found")
                return False
            
            instrument = self.instruments[symbol]
            
            # Update allowed fields
            allowed_fields = [
                'cost_bps', 'effectiveness', 'liquidity_score',
                'min_size', 'max_size', 'delta', 'gamma', 'vega'
            ]
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(instrument, field):
                    setattr(instrument, field, value)
                    logger.info(f"Updated {symbol}.{field} to {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating instrument: {e}")
            return False
