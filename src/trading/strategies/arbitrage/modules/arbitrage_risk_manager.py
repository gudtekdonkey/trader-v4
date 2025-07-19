"""Risk management and validation for arbitrage strategies"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class ArbitrageRiskManager:
    """Manages risk for arbitrage strategies"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.circuit_breaker = {
            'enabled': False,
            'trigger_time': None,
            'duration': 300  # 5 minutes
        }
    
    def validate_market_data(self, market_data: Dict[str, Dict]) -> bool:
        """Validate market data integrity"""
        if not market_data:
            return False
        
        required_fields = ['best_bid', 'best_ask', 'mid_price']
        
        for symbol, data in market_data.items():
            if not isinstance(data, dict):
                return False
            
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing {field} for {symbol}")
                    return False
                
                value = data[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    logger.warning(f"Invalid {field} value for {symbol}: {value}")
                    return False
            
            # Check spread sanity
            spread = data['best_ask'] - data['best_bid']
            if spread < 0:
                logger.error(f"Negative spread for {symbol}")
                return False
            
            spread_pct = spread / data['mid_price']
            if spread_pct > 0.1:  # 10% spread is suspicious
                logger.warning(f"Suspiciously wide spread for {symbol}: {spread_pct:.2%}")
                return False
        
        return True
    
    def validate_opportunity(self, opp) -> bool:
        """Validate arbitrage opportunity"""
        try:
            # Check basic fields
            if not opp.type or not opp.symbols or not opp.entry_prices:
                return False
            
            # Check profit thresholds
            if opp.expected_profit_pct < self.params['min_profit_threshold']:
                return False
            
            if opp.confidence < self.params['min_confidence']:
                return False
            
            # Check execution time
            if opp.execution_time_estimate > self.params['max_execution_time']:
                return False
            
            # Validate prices
            for symbol, price in opp.entry_prices.items():
                if not isinstance(price, (int, float)) or price <= 0:
                    logger.warning(f"Invalid price for {symbol}: {price}")
                    return False
            
            # Validate sizes
            for symbol, size in opp.sizes.items():
                if not isinstance(size, (int, float)) or size < 0:
                    logger.warning(f"Invalid size for {symbol}: {size}")
                    return False
            
            # Check if symbols are blacklisted due to failures
            for symbol in opp.symbols:
                if self.failed_attempts.get(symbol, 0) >= self.max_failed_attempts:
                    logger.warning(f"Symbol {symbol} blacklisted due to failures")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating opportunity: {e}")
            return False
    
    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active"""
        if not self.circuit_breaker['enabled']:
            return False
        
        if self.circuit_breaker['trigger_time']:
            elapsed = pd.Timestamp.now() - self.circuit_breaker['trigger_time']
            if elapsed.total_seconds() < self.circuit_breaker['duration']:
                return True
            else:
                # Reset circuit breaker
                self.circuit_breaker['enabled'] = False
                self.circuit_breaker['trigger_time'] = None
                logger.info("Circuit breaker reset")
        
        return False
    
    def trigger_circuit_breaker(self):
        """Trigger circuit breaker to pause trading"""
        self.circuit_breaker['enabled'] = True
        self.circuit_breaker['trigger_time'] = pd.Timestamp.now()
        logger.warning("Circuit breaker triggered - pausing arbitrage trading")
    
    def record_failure(self, symbols: List[str]):
        """Record failed arbitrage attempt"""
        for symbol in symbols:
            self.failed_attempts[symbol] = self.failed_attempts.get(symbol, 0) + 1
            
            if self.failed_attempts[symbol] >= self.max_failed_attempts:
                logger.warning(f"Symbol {symbol} has reached max failed attempts ({self.max_failed_attempts})")
    
    def reset_failures(self, symbols: List[str]):
        """Reset failure counter for successful arbitrage"""
        for symbol in symbols:
            if symbol in self.failed_attempts:
                self.failed_attempts[symbol] = 0
    
    def check_exposure_limits(self, new_exposure: float, current_exposure: float, 
                            available_capital: float) -> bool:
        """Check if new exposure would exceed limits"""
        max_exposure = self.params.get('max_arbitrage_exposure', 0.2) * available_capital
        
        if current_exposure + new_exposure > max_exposure:
            logger.warning(f"Would exceed max exposure: {current_exposure + new_exposure} > {max_exposure}")
            return False
        
        # Check per-opportunity limit
        max_per_opportunity = self.params.get('max_position_size', 0.1) * available_capital
        if new_exposure > max_per_opportunity:
            logger.warning(f"Single opportunity too large: {new_exposure} > {max_per_opportunity}")
            return False
        
        return True
    
    def calculate_position_sizes(self, opportunity, available_capital: float) -> Dict[str, float]:
        """Calculate safe position sizes for arbitrage"""
        base_sizes = opportunity.sizes.copy()
        
        # Calculate total exposure
        total_exposure = sum(
            abs(size) * opportunity.entry_prices.get(symbol, 0)
            for symbol, size in base_sizes.items()
        )
        
        # Apply position size limit
        max_size = self.params.get('max_position_size', 0.1) * available_capital
        
        if total_exposure > max_size:
            # Scale down all sizes proportionally
            scale_factor = max_size / total_exposure
            
            for symbol in base_sizes:
                base_sizes[symbol] *= scale_factor
            
            logger.info(f"Scaled down position sizes by {scale_factor:.2f}")
        
        return base_sizes
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'circuit_breaker_active': self.is_circuit_breaker_active(),
            'failed_symbols': {
                symbol: count for symbol, count in self.failed_attempts.items()
                if count > 0
            },
            'blacklisted_symbols': [
                symbol for symbol, count in self.failed_attempts.items()
                if count >= self.max_failed_attempts
            ]
        }
