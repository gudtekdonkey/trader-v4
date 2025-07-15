"""
Regime-Aware Position Sizing
Adjusts position sizes based on market regime and conditions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RegimeAwarePositionSizer:
    """Enhanced position sizing that adapts to market regimes"""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        
        # Regime-specific multipliers
        self.regime_multipliers = {
            'trending': {
                'volatility_low': 1.2,
                'volatility_medium': 1.0,
                'volatility_high': 0.7
            },
            'sideways': {
                'volatility_low': 0.8,
                'volatility_medium': 0.6,
                'volatility_high': 0.4
            },
            'volatile': {
                'volatility_low': 0.9,
                'volatility_medium': 0.5,
                'volatility_high': 0.3
            }
        }
        
        # Confidence multipliers
        self.confidence_multipliers = {
            0.9: 1.3,  # Very high confidence
            0.8: 1.2,  # High confidence
            0.7: 1.0,  # Medium confidence
            0.6: 0.8,  # Low confidence
            0.5: 0.6   # Very low confidence
        }
    
    def calculate_regime_aware_position_size(self,
                                           base_size: float,
                                           regime_info: Dict,
                                           signal_strength: float,
                                           confidence: float,
                                           market_conditions: Dict) -> float:
        """
        Calculate position size adjusted for market regime
        
        Args:
            base_size: Base position size from standard calculator
            regime_info: Market regime information
            signal_strength: Signal strength (0-1)
            confidence: Signal confidence (0-1)
            market_conditions: Current market conditions
            
        Returns:
            Adjusted position size
        """
        try:
            # Start with base size
            adjusted_size = base_size
            
            # Apply regime multiplier
            regime_multiplier = self._get_regime_multiplier(regime_info, market_conditions)
            adjusted_size *= regime_multiplier
            
            # Apply confidence multiplier
            confidence_multiplier = self._get_confidence_multiplier(confidence)
            adjusted_size *= confidence_multiplier
            
            # Apply signal strength multiplier
            signal_multiplier = 0.5 + (signal_strength * 0.5)  # Range: 0.5 to 1.0
            adjusted_size *= signal_multiplier
            
            # Apply market stress adjustment
            stress_multiplier = self._get_stress_multiplier(market_conditions)
            adjusted_size *= stress_multiplier
            
            # Apply correlation adjustment
            correlation_multiplier = self._get_correlation_multiplier(market_conditions)
            adjusted_size *= correlation_multiplier
            
            # Ensure position size respects limits
            adjusted_size = self._apply_position_limits(adjusted_size)
            
            logger.debug(f\"Position sizing: base={base_size:.4f}, regime={regime_multiplier:.2f}, \"\n                       f\"confidence={confidence_multiplier:.2f}, signal={signal_multiplier:.2f}, \"\n                       f\"stress={stress_multiplier:.2f}, correlation={correlation_multiplier:.2f}, \"\n                       f\"final={adjusted_size:.4f}\")\n            \n            return adjusted_size\n            \n        except Exception as e:\n            logger.error(f\"Error in regime-aware position sizing: {e}\")\n            return base_size  # Return base size as fallback\n    \n    def _get_regime_multiplier(self, regime_info: Dict, market_conditions: Dict) -> float:\n        \"\"\"Get position size multiplier based on market regime\"\"\"\n        try:\n            # Determine regime type\n            regime_type = regime_info.get('regime_type', 'sideways')\n            if regime_type not in self.regime_multipliers:\n                regime_type = 'sideways'\n            \n            # Determine volatility level\n            volatility = market_conditions.get('volatility', 0.02)\n            if volatility < 0.15:\n                vol_level = 'volatility_low'\n            elif volatility < 0.25:\n                vol_level = 'volatility_medium'\n            else:\n                vol_level = 'volatility_high'\n            \n            multiplier = self.regime_multipliers[regime_type][vol_level]\n            \n            # Additional adjustments based on trend strength\n            trend_strength = regime_info.get('trend_strength', 0.0)\n            if regime_type == 'trending' and trend_strength > 0.7:\n                multiplier *= 1.1  # Boost for strong trends\n            elif regime_type == 'sideways' and trend_strength < 0.3:\n                multiplier *= 0.9  # Reduce for weak trends in sideways market\n            \n            return multiplier\n            \n        except Exception as e:\n            logger.error(f\"Error calculating regime multiplier: {e}\")\n            return 1.0\n    \n    def _get_confidence_multiplier(self, confidence: float) -> float:\n        \"\"\"Get multiplier based on signal confidence\"\"\"\n        try:\n            # Find closest confidence level\n            confidence_levels = sorted(self.confidence_multipliers.keys(), reverse=True)\n            \n            for level in confidence_levels:\n                if confidence >= level:\n                    return self.confidence_multipliers[level]\n            \n            # Default to lowest confidence\n            return self.confidence_multipliers[0.5]\n            \n        except Exception as e:\n            logger.error(f\"Error calculating confidence multiplier: {e}\")\n            return 1.0\n    \n    def _get_stress_multiplier(self, market_conditions: Dict) -> float:\n        \"\"\"Get multiplier based on market stress\"\"\"\n        try:\n            market_stress = market_conditions.get('market_stress', 0.0)\n            \n            # Reduce position size during high stress\n            if market_stress > 0.8:\n                return 0.5  # High stress\n            elif market_stress > 0.6:\n                return 0.7  # Medium stress\n            elif market_stress > 0.4:\n                return 0.9  # Low stress\n            else:\n                return 1.0  # Normal conditions\n                \n        except Exception as e:\n            logger.error(f\"Error calculating stress multiplier: {e}\")\n            return 1.0\n    \n    def _get_correlation_multiplier(self, market_conditions: Dict) -> float:\n        \"\"\"Get multiplier based on portfolio correlation\"\"\"\n        try:\n            market_correlation = market_conditions.get('market_correlation', 0.5)\n            \n            # Reduce position size when correlations are high\n            if market_correlation > 0.8:\n                return 0.7  # High correlation - reduce diversification benefit\n            elif market_correlation > 0.6:\n                return 0.85  # Medium correlation\n            else:\n                return 1.0  # Low correlation - normal sizing\n                \n        except Exception as e:\n            logger.error(f\"Error calculating correlation multiplier: {e}\")\n            return 1.0\n    \n    def _apply_position_limits(self, size: float) -> float:\n        \"\"\"Apply position size limits\"\"\"\n        try:\n            # Get risk parameters\n            risk_params = self.risk_manager.risk_params\n            \n            # Maximum position size (% of capital)\n            max_position_pct = risk_params.get('max_position_size', 0.15)\n            current_capital = self.risk_manager.current_capital\n            max_position_value = current_capital * max_position_pct\n            \n            # Assume average price of $50k for sizing (would use actual price in practice)\n            avg_price = 50000\n            max_size = max_position_value / avg_price\n            \n            # Apply maximum\n            size = min(size, max_size)\n            \n            # Minimum position size\n            min_size = risk_params.get('min_position_size', 0.001)\n            size = max(size, min_size) if size > 0 else 0\n            \n            return size\n            \n        except Exception as e:\n            logger.error(f\"Error applying position limits: {e}\")\n            return size\n    \n    def calculate_kelly_adjusted_size(self,\n                                    base_size: float,\n                                    win_probability: float,\n                                    avg_win: float,\n                                    avg_loss: float) -> float:\n        \"\"\"Calculate Kelly criterion adjusted position size\"\"\"\n        try:\n            if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:\n                return base_size\n            \n            # Kelly formula: f = (bp - q) / b\n            # where b = avg_win/avg_loss, p = win_probability, q = 1-p\n            b = avg_win / abs(avg_loss)\n            p = win_probability\n            q = 1 - p\n            \n            kelly_fraction = (b * p - q) / b\n            \n            # Cap Kelly fraction to reasonable limits\n            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25%\n            \n            # Apply Kelly fraction to base size\n            kelly_size = base_size * kelly_fraction / 0.25  # Normalize\n            \n            return kelly_size\n            \n        except Exception as e:\n            logger.error(f\"Error calculating Kelly adjusted size: {e}\")\n            return base_size\n    \n    def calculate_volatility_adjusted_size(self,\n                                         base_size: float,\n                                         current_volatility: float,\n                                         target_volatility: float = 0.15) -> float:\n        \"\"\"Adjust position size for volatility targeting\"\"\"\n        try:\n            if current_volatility <= 0:\n                return base_size\n            \n            # Volatility scaling factor\n            vol_scaling = target_volatility / current_volatility\n            \n            # Cap scaling to reasonable range\n            vol_scaling = max(0.2, min(vol_scaling, 3.0))\n            \n            return base_size * vol_scaling\n            \n        except Exception as e:\n            logger.error(f\"Error calculating volatility adjusted size: {e}\")\n            return base_size\n    \n    def get_position_sizing_metrics(self) -> Dict:\n        \"\"\"Get current position sizing metrics and statistics\"\"\"\n        try:\n            positions = self.risk_manager.positions\n            current_capital = self.risk_manager.current_capital\n            \n            if not positions:\n                return {}\n            \n            # Calculate position statistics\n            position_values = []\n            position_weights = []\n            \n            for symbol, position in positions.items():\n                current_price = position.get('current_price', position['entry_price'])\n                value = position['size'] * current_price\n                weight = value / current_capital if current_capital > 0 else 0\n                \n                position_values.append(value)\n                position_weights.append(weight)\n            \n            metrics = {\n                'total_positions': len(positions),\n                'total_position_value': sum(position_values),\n                'capital_utilization': sum(position_values) / current_capital if current_capital > 0 else 0,\n                'average_position_weight': np.mean(position_weights) if position_weights else 0,\n                'max_position_weight': max(position_weights) if position_weights else 0,\n                'min_position_weight': min(position_weights) if position_weights else 0,\n                'position_weight_std': np.std(position_weights) if position_weights else 0,\n                'concentration_ratio': max(position_weights) / sum(position_weights) if position_weights else 0\n            }\n            \n            return metrics\n            \n        except Exception as e:\n            logger.error(f\"Error calculating position sizing metrics: {e}\")\n            return {}\n    \n    def suggest_optimal_position_count(self, \n                                     available_capital: float,\n                                     avg_position_size: float,\n                                     correlation_matrix: Optional[pd.DataFrame] = None) -> int:\n        \"\"\"Suggest optimal number of positions for diversification\"\"\"\n        try:\n            # Basic calculation based on capital\n            basic_count = int(available_capital / avg_position_size)\n            \n            # Adjust for correlation if available\n            if correlation_matrix is not None:\n                # Higher correlation means need more positions for diversification\n                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()\n                \n                if avg_correlation > 0.7:\n                    correlation_adjustment = 1.5  # Need more positions\n                elif avg_correlation > 0.5:\n                    correlation_adjustment = 1.2\n                else:\n                    correlation_adjustment = 1.0\n                \n                basic_count = int(basic_count * correlation_adjustment)\n            \n            # Apply practical limits\n            optimal_count = max(3, min(basic_count, 15))  # Between 3 and 15 positions\n            \n            return optimal_count\n            \n        except Exception as e:\n            logger.error(f\"Error suggesting optimal position count: {e}\")\n            return 5  # Default\n    \n    def update_regime_parameters(self, new_params: Dict):\n        \"\"\"Update regime-specific parameters\"\"\"\n        try:\n            if 'regime_multipliers' in new_params:\n                self.regime_multipliers.update(new_params['regime_multipliers'])\n            \n            if 'confidence_multipliers' in new_params:\n                self.confidence_multipliers.update(new_params['confidence_multipliers'])\n            \n            logger.info(\"Regime parameters updated\")\n            \n        except Exception as e:\n            logger.error(f\"Error updating regime parameters: {e}\")\n\n# Example usage and testing\nif __name__ == '__main__':\n    # Mock risk manager for testing\n    class MockRiskManager:\n        def __init__(self):\n            self.current_capital = 100000\n            self.risk_params = {\n                'max_position_size': 0.15,\n                'min_position_size': 0.001\n            }\n            self.positions = {\n                'BTC-USD': {\n                    'size': 0.5,\n                    'entry_price': 45000,\n                    'current_price': 47000,\n                    'side': 'long'\n                }\n            }\n    \n    # Test regime-aware position sizer\n    risk_manager = MockRiskManager()\n    sizer = RegimeAwarePositionSizer(risk_manager)\n    \n    # Test position sizing\n    base_size = 0.1\n    regime_info = {\n        'regime_type': 'trending',\n        'trend_strength': 0.8\n    }\n    market_conditions = {\n        'volatility': 0.18,\n        'market_stress': 0.3,\n        'market_correlation': 0.6\n    }\n    \n    adjusted_size = sizer.calculate_regime_aware_position_size(\n        base_size=base_size,\n        regime_info=regime_info,\n        signal_strength=0.7,\n        confidence=0.8,\n        market_conditions=market_conditions\n    )\n    \n    print(f\"Base size: {base_size}\")\n    print(f\"Adjusted size: {adjusted_size:.4f}\")\n    print(f\"Sizing metrics: {sizer.get_position_sizing_metrics()}\")\n