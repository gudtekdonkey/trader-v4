"""
Risk control module for market making strategy.
Handles circuit breakers, emergency stops, and risk parameter validation.
"""

import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RiskController:
    """Controls risk parameters and implements safety mechanisms"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.emergency_stop = False
        self.circuit_breaker = {
            'consecutive_failures': 0,
            'last_failure_time': None,
            'cooldown_period': 60  # seconds
        }
        self.quote_failures = {}
        self.risk_events = []
        
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        try:
            # Spread validation
            assert 0 < self.params['min_spread'] < self.params['max_spread'] < 0.1, \
                "Invalid spread parameters"
            
            # Inventory validation
            assert self.params['max_inventory'] > 0, "Invalid max inventory"
            assert 0 <= self.params['inventory_skew_factor'] <= 1, "Invalid skew factor"
            
            # Order validation
            assert self.params['order_levels'] > 0, "Invalid order levels"
            assert self.params['min_order_size'] > 0, "Invalid minimum order size"
            assert 0 < self.params['size_decay'] < 1, "Invalid size decay"
            
            # Risk parameters
            assert 0 < self.params['rebalance_threshold'] <= 1, "Invalid rebalance threshold"
            assert self.params['max_quote_failures'] > 0, "Invalid max quote failures"
            assert self.params['emergency_spread_multiplier'] > 1, "Invalid emergency multiplier"
            
            return True
            
        except AssertionError as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is active"""
        if self.circuit_breaker['consecutive_failures'] >= self.params['max_quote_failures']:
            if self.circuit_breaker['last_failure_time']:
                elapsed = (pd.Timestamp.now() - self.circuit_breaker['last_failure_time']).seconds
                if elapsed < self.circuit_breaker['cooldown_period']:
                    return True
                else:
                    # Reset circuit breaker
                    self.reset_circuit_breaker()
        return False
    
    def record_quote_failure(self, symbol: str):
        """Record a quote generation failure"""
        self.quote_failures[symbol] = self.quote_failures.get(symbol, 0) + 1
        self.circuit_breaker['consecutive_failures'] += 1
        self.circuit_breaker['last_failure_time'] = pd.Timestamp.now()
        
        # Log risk event
        self.risk_events.append({
            'timestamp': pd.Timestamp.now(),
            'type': 'quote_failure',
            'symbol': symbol,
            'consecutive_failures': self.circuit_breaker['consecutive_failures']
        })
        
        # Check for emergency stop
        if self.circuit_breaker['consecutive_failures'] >= self.params['max_quote_failures']:
            logger.error(f"Max quote failures reached for {symbol}, activating emergency stop")
            self.activate_emergency_stop('max_quote_failures')
    
    def record_quote_success(self):
        """Record successful quote generation"""
        if self.circuit_breaker['consecutive_failures'] > 0:
            self.circuit_breaker['consecutive_failures'] = 0
            logger.info("Quote generation successful, reset failure counter")
    
    def check_inventory_limits(self, symbol: str, inventory_value: float) -> bool:
        """Check if inventory is within risk limits"""
        try:
            if abs(inventory_value) > self.params['max_inventory']:
                logger.warning(f"Inventory limit breached for {symbol}: {inventory_value}")
                self.risk_events.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'inventory_limit_breach',
                    'symbol': symbol,
                    'inventory_value': inventory_value
                })
                return False
            
            # Check for rapid inventory change
            recent_events = [e for e in self.risk_events 
                           if e['symbol'] == symbol and 
                           e['type'] == 'inventory_change' and
                           (pd.Timestamp.now() - e['timestamp']).seconds < 300]
            
            if len(recent_events) > 10:  # Too many inventory changes
                logger.warning(f"Rapid inventory changes detected for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking inventory limits: {e}")
            return False
    
    def validate_market_conditions(self, market_data: Dict) -> bool:
        """Validate market conditions for safe trading"""
        try:
            # Check for valid prices
            if not self._validate_market_data(market_data):
                return False
            
            # Check spread
            spread = market_data['best_ask'] - market_data['best_bid']
            mid_price = (market_data['best_ask'] + market_data['best_bid']) / 2
            spread_pct = spread / mid_price
            
            # Check for abnormal spread
            if spread_pct > 0.1:  # 10% spread
                logger.warning(f"Abnormal spread detected: {spread_pct:.2%}")
                self.risk_events.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'abnormal_spread',
                    'symbol': market_data.get('symbol', 'UNKNOWN'),
                    'spread_pct': spread_pct
                })
                return False
            
            # Check volatility if available
            if 'volatility' in market_data:
                volatility = market_data['volatility']
                if volatility > 0.5:  # 50% volatility
                    logger.warning(f"High volatility detected: {volatility:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating market conditions: {e}")
            return False
    
    def _validate_market_data(self, market_data: Dict) -> bool:
        """Validate market data integrity"""
        required_fields = ['best_bid', 'best_ask', 'symbol']
        
        for field in required_fields:
            if field not in market_data:
                return False
        
        # Validate prices
        for field in ['best_bid', 'best_ask']:
            value = market_data.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                return False
        
        # Check spread
        if market_data['best_ask'] <= market_data['best_bid']:
            return False
        
        return True
    
    def calculate_base_size(self, available_capital: float, market_data: Dict, 
                          level: int) -> float:
        """Calculate base order size with risk limits"""
        try:
            if available_capital <= 0:
                logger.warning("No available capital")
                return 0
            
            # Base allocation per position
            max_position_size = min(
                available_capital * 0.1,  # 10% per position
                self.params['max_inventory'] * 0.2  # 20% of max inventory
            )
            
            # Adjust for market conditions
            volatility = market_data.get('volatility', 0.02)
            volatility_factor = max(0.3, 1 - volatility * 10)  # Reduce size in high volatility
            
            # Level-based sizing
            level_factor = 1 / (level + 1)
            
            base_size = max_position_size * volatility_factor * level_factor
            
            # Ensure minimum
            current_price = (market_data['best_bid'] + market_data['best_ask']) / 2
            min_size_value = self.params['min_order_size']
            min_size = min_size_value / current_price
            
            return max(base_size, min_size)
            
        except Exception as e:
            logger.error(f"Error calculating base size: {e}")
            return self.params['min_order_size'] / 50000  # Fallback assuming BTC price
    
    def activate_emergency_stop(self, reason: str):
        """Activate emergency stop"""
        logger.error(f"EMERGENCY STOP ACTIVATED: {reason}")
        self.emergency_stop = True
        
        self.risk_events.append({
            'timestamp': pd.Timestamp.now(),
            'type': 'emergency_stop',
            'reason': reason
        })
    
    def reset_emergency_stop(self):
        """Reset emergency stop with validation"""
        if self.emergency_stop:
            logger.info("Resetting emergency stop")
            self.emergency_stop = False
            self.reset_circuit_breaker()
            
            self.risk_events.append({
                'timestamp': pd.Timestamp.now(),
                'type': 'emergency_stop_reset'
            })
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.circuit_breaker['consecutive_failures'] = 0
        self.circuit_breaker['last_failure_time'] = None
        logger.info("Circuit breaker reset")
    
    def get_risk_status(self) -> Dict:
        """Get current risk control status"""
        return {
            'emergency_stop': self.emergency_stop,
            'circuit_breaker_active': self.check_circuit_breaker(),
            'consecutive_failures': self.circuit_breaker['consecutive_failures'],
            'failed_symbols': list(self.quote_failures.keys()),
            'recent_risk_events': self._get_recent_risk_events()
        }
    
    def _get_recent_risk_events(self, hours: int = 1) -> List[Dict]:
        """Get recent risk events"""
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        recent_events = []
        
        for event in self.risk_events[-100:]:  # Last 100 events
            if event['timestamp'] > cutoff_time:
                event_copy = event.copy()
                event_copy['timestamp'] = event_copy['timestamp'].isoformat()
                recent_events.append(event_copy)
        
        return recent_events
    
    def should_reduce_risk(self) -> bool:
        """Determine if risk should be reduced"""
        # Check multiple risk indicators
        if self.emergency_stop:
            return True
        
        if self.check_circuit_breaker():
            return True
        
        # Check recent risk events
        recent_events = self._get_recent_risk_events(hours=1)
        if len(recent_events) > 20:  # Too many risk events
            return True
        
        # Check failure rate
        if self.circuit_breaker['consecutive_failures'] > self.params['max_quote_failures'] / 2:
            return True
        
        return False
    
    def get_risk_adjustment_factor(self) -> float:
        """Get risk adjustment factor for position sizing"""
        if self.emergency_stop:
            return 0  # No trading
        
        if self.check_circuit_breaker():
            return 0.1  # Minimal trading
        
        # Scale based on consecutive failures
        failure_ratio = self.circuit_breaker['consecutive_failures'] / self.params['max_quote_failures']
        adjustment = 1 - (failure_ratio * 0.8)  # Reduce up to 80%
        
        return max(0.2, adjustment)