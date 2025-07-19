"""
Market regime detection
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regime changes"""
    
    def detect_regime_changes(self, 
                            returns: pd.Series, 
                            window: int = 60) -> Dict:
        """Detect market regime changes that might affect portfolio allocation"""
        try:
            regime_indicators = {}
            
            # Rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            current_vol = rolling_vol.iloc[-1]
            avg_vol = rolling_vol.mean()
            vol_regime = self._classify_volatility_regime(current_vol, avg_vol)
            
            # Rolling correlation with trend
            rolling_returns = returns.rolling(window).mean() * 252
            trend_strength = abs(rolling_returns.iloc[-1])
            
            # Market stress indicator (based on drawdowns)
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.rolling(window).max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            current_drawdown = abs(drawdowns.iloc[-1])
            
            stress_level = self._classify_stress_level(current_drawdown)
            
            regime_indicators = {
                'volatility_regime': vol_regime,
                'current_volatility': current_vol,
                'average_volatility': avg_vol,
                'trend_strength': trend_strength,
                'stress_level': stress_level,
                'current_drawdown': current_drawdown,
                'rebalancing_urgency': self._determine_rebalancing_urgency(stress_level, vol_regime)
            }
            
            return regime_indicators
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {e}")
            return {}
    
    def _classify_volatility_regime(self, current_vol: float, avg_vol: float) -> str:
        """Classify volatility regime"""
        if current_vol > avg_vol * 1.2:
            return 'high'
        elif current_vol < avg_vol * 0.8:
            return 'low'
        else:
            return 'normal'
    
    def _classify_stress_level(self, current_drawdown: float) -> str:
        """Classify market stress level"""
        if current_drawdown > 0.1:
            return 'high'
        elif current_drawdown > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def _determine_rebalancing_urgency(self, stress_level: str, vol_regime: str) -> str:
        """Determine rebalancing urgency based on regime"""
        if stress_level == 'high' or vol_regime == 'high':
            return 'high'
        else:
            return 'normal'
