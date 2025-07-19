"""
Quote generation module for market making strategy.
Handles fair value calculation, spread determination, and quote creation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketMakingQuote:
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    spread: float
    mid_price: float
    fair_value: float
    inventory_skew: float
    volatility_adjustment: float
    confidence: float


class QuoteGenerator:
    """Generates market making quotes based on market conditions and strategy parameters"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.circuit_breaker = {
            'consecutive_failures': 0,
            'last_failure_time': None,
            'cooldown_period': 60  # seconds
        }
        self.quote_failures = {}
        
    def calculate_fair_value(self, market_data: Dict, ml_predictions: Optional[Dict] = None) -> float:
        """Calculate fair value with multiple signal sources"""
        try:
            # Start with market mid price
            best_bid = market_data['best_bid']
            best_ask = market_data['best_ask']
            market_mid = (best_bid + best_ask) / 2
            
            fair_value_components = []
            weights = []
            
            # Market mid price (always included)
            fair_value_components.append(market_mid)
            weights.append(0.4)
            
            # Order flow imbalance signal
            self._add_order_imbalance_signal(market_data, market_mid, fair_value_components, weights)
            
            # Recent trades signal
            self._add_recent_trades_signal(market_data, market_mid, fair_value_components, weights)
            
            # ML prediction signal
            self._add_ml_prediction_signal(ml_predictions, market_mid, fair_value_components, weights)
            
            # Microprice
            self._add_microprice_signal(market_data, best_bid, best_ask, market_mid, 
                                       fair_value_components, weights)
            
            # Calculate weighted fair value
            fair_value = self._calculate_weighted_average(fair_value_components, weights, market_mid)
            
            # Final sanity check
            if not (0.5 * market_mid < fair_value < 2 * market_mid):
                logger.warning(f"Fair value out of range: {fair_value}, using market mid")
                fair_value = market_mid
            
            return fair_value
            
        except Exception as e:
            logger.error(f"Error calculating fair value: {e}")
            # Fallback to simple mid price
            return (market_data['best_bid'] + market_data['best_ask']) / 2
    
    def _add_order_imbalance_signal(self, market_data: Dict, market_mid: float, 
                                   components: List, weights: List):
        """Add order imbalance signal to fair value calculation"""
        try:
            if 'order_imbalance' in market_data:
                imbalance = market_data['order_imbalance']
                if -1 <= imbalance <= 1:  # Validate range
                    flow_adjustment = market_mid * imbalance * 0.001
                    components.append(market_mid + flow_adjustment)
                    weights.append(0.2)
        except Exception as e:
            logger.debug(f"Error processing order imbalance: {e}")
    
    def _add_recent_trades_signal(self, market_data: Dict, market_mid: float,
                                 components: List, weights: List):
        """Add recent trades VWAP signal"""
        try:
            if 'recent_trades' in market_data:
                trades = market_data['recent_trades']
                if trades and isinstance(trades, list):
                    # VWAP of recent trades
                    valid_trades = [t for t in trades if isinstance(t, dict) and 
                                  'price' in t and 'size' in t and
                                  t['price'] > 0 and t['size'] > 0]
                    
                    if valid_trades:
                        total_volume = sum(t['size'] for t in valid_trades)
                        if total_volume > 0:
                            vwap = sum(t['price'] * t['size'] for t in valid_trades) / total_volume
                            if 0.5 * market_mid < vwap < 2 * market_mid:  # Sanity check
                                components.append(vwap)
                                weights.append(0.2)
        except Exception as e:
            logger.debug(f"Error processing recent trades: {e}")
    
    def _add_ml_prediction_signal(self, ml_predictions: Optional[Dict], market_mid: float,
                                components: List, weights: List):
        """Add ML prediction signal"""
        try:
            if ml_predictions and 'price_prediction' in ml_predictions:
                ml_price = ml_predictions['price_prediction']
                ml_confidence = ml_predictions.get('confidence', 0.5)
                
                # Validate ML prediction
                if (isinstance(ml_price, (int, float)) and ml_price > 0 and
                    0.5 * market_mid < ml_price < 2 * market_mid and
                    0 <= ml_confidence <= 1):
                    components.append(ml_price)
                    weights.append(0.2 * ml_confidence)
        except Exception as e:
            logger.debug(f"Error processing ML predictions: {e}")
    
    def _add_microprice_signal(self, market_data: Dict, best_bid: float, best_ask: float,
                              market_mid: float, components: List, weights: List):
        """Add microprice signal based on order book volumes"""
        try:
            if 'bid_volume' in market_data and 'ask_volume' in market_data:
                bid_vol = market_data['bid_volume']
                ask_vol = market_data['ask_volume']
                if bid_vol > 0 and ask_vol > 0:
                    microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
                    if 0.9 * market_mid < microprice < 1.1 * market_mid:  # Sanity check
                        components.append(microprice)
                        weights.append(0.2)
        except Exception as e:
            logger.debug(f"Error calculating microprice: {e}")
    
    def _calculate_weighted_average(self, components: List, weights: List, default: float) -> float:
        """Calculate weighted average with error handling"""
        if len(components) > 0 and len(weights) == len(components):
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                return sum(comp * weight for comp, weight in zip(components, weights))
        return default
    
    def calculate_spread(self, level: int, market_spread: float, volatility: float,
                        inventory_skew: float) -> float:
        """Calculate spread for a specific quote level"""
        try:
            # Base spread increases with level
            base = self.params['base_spread'] * (1 + level * 0.5)
            
            # Volatility adjustment
            volatility_spread = base * (1 + volatility * self.params['volatility_factor'])
            
            # Competition adjustment
            if market_spread > 0:
                competition_factor = min(1, market_spread / self.params['base_spread'])
                adjusted_spread = volatility_spread * (0.5 + 0.5 * competition_factor)
            else:
                adjusted_spread = volatility_spread
            
            # Apply emergency mode if needed
            if self.circuit_breaker['consecutive_failures'] > 5:
                adjusted_spread *= self.params['emergency_spread_multiplier']
            
            # Ensure within bounds
            final_spread = max(self.params['min_spread'], 
                             min(self.params['max_spread'], adjusted_spread))
            
            return final_spread
            
        except Exception as e:
            logger.error(f"Error calculating level spread: {e}")
            return self.params['base_spread']
    
    def generate_quote(self, level: int, symbol: str, fair_value: float,
                      current_spread: float, volatility: float, inventory_skew: float,
                      base_size: float, market_data: Dict,
                      ml_predictions: Optional[Dict] = None) -> Optional[MarketMakingQuote]:
        """Generate a single quote for a specific level"""
        try:
            # Calculate spread for this level
            level_spread = self.calculate_spread(level, current_spread, volatility, inventory_skew)
            
            # Calculate prices
            bid_price = fair_value - level_spread / 2 - (level * self.params['level_spacing'] * fair_value)
            ask_price = fair_value + level_spread / 2 + (level * self.params['level_spacing'] * fair_value)
            
            # Apply inventory skew
            skew_adjustment = inventory_skew * level_spread * self.params['inventory_skew_factor']
            bid_price -= skew_adjustment
            ask_price -= skew_adjustment
            
            # Ensure positive prices
            bid_price = max(bid_price, fair_value * 0.5)  # At least 50% of fair value
            ask_price = max(ask_price, bid_price * 1.001)  # At least 0.1% spread
            
            # Calculate sizes with inventory adjustment
            bid_size = base_size * (1 + max(-0.9, min(0.9, inventory_skew * 0.5)))
            ask_size = base_size * (1 - max(-0.9, min(0.9, inventory_skew * 0.5)))
            
            # Apply size decay
            size_multiplier = max(0.1, (1 - self.params['size_decay']) ** level)
            bid_size *= size_multiplier
            ask_size *= size_multiplier
            
            # Ensure minimum size
            min_size = self.params['min_order_size'] / ((bid_price + ask_price) / 2)
            bid_size = max(bid_size, min_size)
            ask_size = max(ask_size, min_size)
            
            # Calculate confidence
            confidence = self._calculate_quote_confidence(
                market_data, fair_value, level, ml_predictions
            )
            
            return MarketMakingQuote(
                symbol=symbol,
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size,
                spread=ask_price - bid_price,
                mid_price=(bid_price + ask_price) / 2,
                fair_value=fair_value,
                inventory_skew=inventory_skew,
                volatility_adjustment=volatility,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error generating level quote: {e}")
            return None
    
    def _calculate_quote_confidence(self, market_data: Dict, fair_value: float,
                                  level: int, ml_predictions: Optional[Dict]) -> float:
        """Calculate confidence score for a quote"""
        try:
            confidence_factors = []
            
            # Fair value confidence
            mid_price = (market_data['best_bid'] + market_data['best_ask']) / 2
            price_deviation = abs(fair_value - mid_price) / mid_price
            price_confidence = max(0, 1 - price_deviation * 50)
            confidence_factors.append(price_confidence)
            
            # Spread confidence
            spread = market_data['best_ask'] - market_data['best_bid']
            spread_pct = spread / mid_price
            spread_confidence = max(0, 1 - spread_pct * 100)
            confidence_factors.append(spread_confidence)
            
            # Level confidence
            level_confidence = 1 / (level + 1)
            confidence_factors.append(level_confidence)
            
            # ML confidence
            if ml_predictions and 'confidence' in ml_predictions:
                ml_conf = ml_predictions.get('confidence', 0.5)
                if 0 <= ml_conf <= 1:
                    confidence_factors.append(ml_conf)
            
            # Circuit breaker adjustment
            if self.circuit_breaker['consecutive_failures'] > 0:
                failure_penalty = 1 - (self.circuit_breaker['consecutive_failures'] / 10)
                confidence_factors.append(max(0.1, failure_penalty))
            
            return max(0.1, min(0.95, np.mean(confidence_factors)))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def generate_emergency_quote(self, symbol: str, market_data: Dict,
                               current_inventory: float, mid_price: float) -> Optional[MarketMakingQuote]:
        """Generate emergency quote to reduce inventory exposure"""
        try:
            logger.warning(f"Generating emergency quotes for {symbol}")
            
            # Wide spread to avoid further exposure
            emergency_spread = self.params['max_spread'] * self.params['emergency_spread_multiplier']
            
            # Single quote to reduce position
            if current_inventory > 0:
                # Long inventory - aggressive ask, defensive bid
                bid_price = mid_price - emergency_spread
                ask_price = mid_price - emergency_spread * 0.1  # Close to mid to sell
                bid_size = self.params['min_order_size'] / bid_price
                ask_size = abs(current_inventory) * 0.1  # Sell 10% of inventory
            else:
                # Short inventory - aggressive bid, defensive ask
                bid_price = mid_price + emergency_spread * 0.1  # Close to mid to buy
                ask_price = mid_price + emergency_spread
                bid_size = abs(current_inventory) * 0.1  # Buy 10% of inventory
                ask_size = self.params['min_order_size'] / ask_price
            
            return MarketMakingQuote(
                symbol=symbol,
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size,
                spread=ask_price - bid_price,
                mid_price=(bid_price + ask_price) / 2,
                fair_value=mid_price,
                inventory_skew=np.sign(current_inventory),
                volatility_adjustment=1.0,
                confidence=0.3  # Low confidence for emergency quotes
            )
            
        except Exception as e:
            logger.error(f"Error generating emergency quotes: {e}")
            return None
    
    def validate_quote(self, quote: MarketMakingQuote, market_data: Dict) -> bool:
        """Validate generated quote against market conditions"""
        try:
            # Price validation
            if quote.bid_price <= 0 or quote.ask_price <= 0:
                return False
            
            if quote.bid_price >= quote.ask_price:
                return False
            
            # Size validation
            if quote.bid_size <= 0 or quote.ask_size <= 0:
                return False
            
            if quote.bid_size < self.params['min_order_size'] / quote.bid_price:
                return False
            
            if quote.ask_size < self.params['min_order_size'] / quote.ask_price:
                return False
            
            # Spread validation
            if quote.spread < self.params['min_spread'] * quote.mid_price:
                return False
            
            if quote.spread > self.params['max_spread'] * quote.mid_price:
                return False
            
            # Price deviation check - quotes shouldn't be too far from market
            market_mid = (market_data['best_bid'] + market_data['best_ask']) / 2
            max_deviation = 0.05  # 5%
            
            if abs(quote.mid_price - market_mid) / market_mid > max_deviation:
                logger.warning(f"Quote too far from market: {quote.mid_price} vs {market_mid}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating quote: {e}")
            return False