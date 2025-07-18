"""
Advanced sizing methods module for position sizing.
Implements sophisticated position sizing algorithms.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AdvancedSizingMethods:
    """Implements advanced position sizing methods"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
    def optimal_f_size(self, historical_results: List[float], 
                      available_capital: float) -> float:
        """
        Optimal f position sizing based on historical performance with error handling.
        
        Args:
            historical_results: Historical trade results
            available_capital: Available capital
            
        Returns:
            Optimal f based position size
        """
        try:
            if len(historical_results) < 20:
                # Not enough history
                logger.warning("Insufficient history for optimal f")
                return self.params['min_position_size']
            
            # Validate returns
            returns = np.array(historical_results)
            if not np.isfinite(returns).all():
                logger.warning("Invalid historical returns")
                return self.params['min_position_size']
            
            # Grid search for optimal f
            f_values = np.linspace(0.01, 0.5, 50)
            twrs = []
            
            for f in f_values:
                try:
                    twr = self._calculate_twr(returns, f)
                    twrs.append(twr)
                except Exception as e:
                    logger.warning(f"Error calculating TWR for f={f}: {e}")
                    twrs.append(1.0)
            
            # Find optimal f
            if twrs:
                optimal_idx = np.argmax(twrs)
                optimal_f = f_values[optimal_idx]
            else:
                optimal_f = 0.02  # Default
            
            # Apply safety factor
            optimal_f *= 0.5
            
            # Convert to position size
            if available_capital <= 0:
                return self.params['min_position_size']
                
            position_size = available_capital * optimal_f
            
            return max(position_size, self.params['min_position_size'])
            
        except Exception as e:
            logger.error(f"Error in optimal f sizing: {e}")
            return self.params['min_position_size']
    
    def risk_parity_size(self, symbol: str, positions: Dict[str, Any],
                        market_volatility: float) -> float:
        """
        Risk parity position sizing with error handling.
        
        Size positions so each contributes equally to portfolio risk.
        
        Args:
            symbol: Trading symbol
            positions: Current positions
            market_volatility: Market volatility for new position
            
        Returns:
            Risk parity position size
        """
        try:
            if not positions:
                # First position, use default allocation
                return self.params['min_position_size'] * 10
            
            # Calculate risk contribution of each position
            risk_contributions = {}
            total_risk = 0
            
            for sym, pos in positions.items():
                try:
                    volatility = pos.get('volatility', 0.02)
                    current_price = pos.get('current_price', pos.get('entry_price', 0))
                    size = pos.get('size', 0)
                    
                    # Validate values
                    if volatility > 0 and current_price > 0 and size > 0:
                        position_value = size * current_price
                        risk_contribution = position_value * volatility
                        risk_contributions[sym] = risk_contribution
                        total_risk += risk_contribution
                except Exception as e:
                    logger.warning(f"Error calculating risk for {sym}: {e}")
                    continue
            
            if total_risk <= 0:
                return self.params['min_position_size']
            
            # Target equal risk contribution
            target_risk = total_risk / (len(positions) + 1)  # +1 for new position
            
            # Calculate position size for target risk
            if market_volatility > 0:
                position_size = target_risk / market_volatility
            else:
                position_size = self.params['min_position_size'] * 5
            
            return max(position_size, self.params['min_position_size'])
            
        except Exception as e:
            logger.error(f"Error in risk parity sizing: {e}")
            return self.params['min_position_size']
    
    def ml_based_size(self, features: List[float], 
                     available_capital: float) -> float:
        """
        Machine learning-based position sizing with error handling.
        
        Uses features to predict optimal position size.
        
        Args:
            features: Feature vector for ML model
            available_capital: Available capital
            
        Returns:
            ML-predicted position size
        """
        try:
            # Validate features
            features = [f if np.isfinite(f) else 0 for f in features]
            
            # Simple ML-based sizing (placeholder implementation)
            # In practice, would use trained model
            confidence_factor = max(0.1, min(1.0, features[0]))  # First feature is confidence
            volatility_factor = min(0.02 / max(features[1], 0.001), 2.0)  # Second is volatility
            
            if available_capital <= 0:
                return self.params['min_position_size']
                
            base_size = available_capital * 0.1
            ml_adjusted_size = base_size * confidence_factor * volatility_factor
            
            return max(ml_adjusted_size, self.params['min_position_size'])
            
        except Exception as e:
            logger.error(f"Error in ML-based sizing: {e}")
            return self.params['min_position_size']
    
    def regime_adjusted_size(self, regime: str, base_size: float,
                           regime_confidence: float = 0.7) -> float:
        """
        Regime-adjusted position sizing with error handling.
        
        Adjusts size based on detected market regime.
        
        Args:
            regime: Market regime
            base_size: Base position size
            regime_confidence: Confidence in regime detection
            
        Returns:
            Regime-adjusted position size
        """
        try:
            # Validate regime
            if regime not in self.params['regime_adjustments']:
                logger.warning(f"Unknown regime: {regime}, using normal")
                regime = 'normal'
            
            multiplier = self.params['regime_adjustments'].get(regime, 1.0)
            
            # Additional adjustment for regime confidence
            if 0 < regime_confidence < 0.5:
                multiplier = (multiplier + 1.0) / 2  # Average with neutral sizing
            
            adjusted_size = base_size * multiplier
            
            return max(adjusted_size, self.params['min_position_size'])
            
        except Exception as e:
            logger.error(f"Error in regime-adjusted sizing: {e}")
            return base_size if base_size > 0 else self.params['min_position_size']
    
    def _calculate_twr(self, returns: np.ndarray, f: float) -> float:
        """
        Calculate Terminal Wealth Relative for optimal f with error handling.
        
        Args:
            returns: Array of returns
            f: Fraction to test
            
        Returns:
            Terminal wealth relative
        """
        try:
            twr = 1.0
            for r in returns:
                twr *= (1 + f * r)
                if twr <= 0:
                    return 0
                if not np.isfinite(twr):
                    return 0
            return twr
        except Exception as e:
            logger.error(f"Error calculating TWR: {e}")
            return 1.0
    
    @staticmethod
    def get_historical_results(signal: Dict[str, Any]) -> List[float]:
        """
        Get historical results for similar signals with error handling.
        
        Args:
            signal: Trading signal
            
        Returns:
            List of historical returns
        """
        try:
            # In practice, this would query a database of historical trades
            # For now, return simulated results
            
            # Simulate based on signal confidence
            confidence = signal.get('confidence', 0.5)
            confidence = max(0.1, min(0.9, confidence))  # Bound confidence
            
            win_rate = 0.4 + confidence * 0.3  # 40-70% win rate based on confidence
            
            results = []
            for _ in range(50):
                if np.random.random() < win_rate:
                    # Win
                    result = np.random.uniform(0.01, 0.05)  # 1-5% win
                else:
                    # Loss
                    result = np.random.uniform(-0.03, -0.01)  # 1-3% loss
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting historical results: {e}")
            return [0.01] * 20  # Default small positive returns
