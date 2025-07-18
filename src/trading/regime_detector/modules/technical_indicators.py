"""
Technical Indicators Module

Calculates technical indicators used in regime detection
including RSI, ADX, Bollinger Bands, MACD, and ATR.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for market regime detection.
    
    This class provides methods to calculate various technical indicators
    that are used as features in regime detection.
    """
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        try:
            # Validate inputs
            if prices is None or prices.empty:
                return pd.Series(50, index=prices.index if prices is not None else [])
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values
            rsi = rsi.fillna(50)
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=prices.index)
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index.
        
        Args:
            data: DataFrame with high, low, close prices
            period: ADX period
            
        Returns:
            ADX values
        """
        try:
            # Check required columns
            required = ['high', 'low', 'close']
            if not all(col in data.columns for col in required):
                logger.warning("Missing required columns for ADX calculation")
                return pd.Series(25, index=data.index)  # Neutral ADX value
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period, min_periods=1).mean()
            
            # Directional movements
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            pos_dm = up_move.where(
                (up_move > down_move) & (up_move > 0), 0
            )
            neg_dm = down_move.where(
                (down_move > up_move) & (down_move > 0), 0
            )
            
            # Directional indicators
            atr_safe = atr.replace(0, 1e-10)
            pos_di = 100 * (pos_dm.rolling(period, min_periods=1).mean() / atr_safe)
            neg_di = 100 * (neg_dm.rolling(period, min_periods=1).mean() / atr_safe)
            
            # ADX calculation
            di_sum = pos_di + neg_di
            di_sum_safe = di_sum.replace(0, 1e-10)
            dx = 100 * abs(pos_di - neg_di) / di_sum_safe
            adx = dx.rolling(period, min_periods=1).mean()
            
            # Fill NaN values
            adx = adx.fillna(25)  # Neutral ADX
            
            return adx
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return pd.Series(25, index=data.index)
    
    def calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: float = 2
    ) -> List[pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            List of [band_width, price_position]
        """
        try:
            if prices is None or prices.empty:
                return [pd.Series(0, index=prices.index if prices is not None else [])] * 2
            
            # Calculate bands
            sma = prices.rolling(window=period, min_periods=1).mean()
            std = prices.rolling(window=period, min_periods=1).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Band width (normalized)
            band_width = (upper_band - lower_band) / sma
            band_width = band_width.fillna(0)
            
            # Price position within bands (-1 to 1)
            price_position = 2 * (prices - lower_band) / (upper_band - lower_band) - 1
            price_position = price_position.fillna(0)
            price_position = price_position.clip(-1, 1)
            
            return [band_width, price_position]
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return [pd.Series(0, index=prices.index)] * 2
    
    def calculate_macd(
        self, 
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> List[pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            List of [macd_line, macd_histogram]
        """
        try:
            if prices is None or prices.empty:
                return [pd.Series(0, index=prices.index if prices is not None else [])] * 2
            
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # MACD histogram
            macd_histogram = macd_line - signal_line
            
            # Normalize by price level
            macd_line_norm = macd_line / prices
            macd_histogram_norm = macd_histogram / prices
            
            return [macd_line_norm.fillna(0), macd_histogram_norm.fillna(0)]
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return [pd.Series(0, index=prices.index)] * 2
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            data: DataFrame with high, low, close prices
            period: ATR period
            
        Returns:
            ATR values
        """
        try:
            # Check required columns
            required = ['high', 'low', 'close']
            if not all(col in data.columns for col in required):
                logger.warning("Missing required columns for ATR calculation")
                return pd.Series(0, index=data.index)
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR
            atr = tr.rolling(window=period, min_periods=1).mean()
            
            # Normalize by price
            atr_pct = atr / close
            
            return atr_pct.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(0, index=data.index)
    
    def calculate_momentum(
        self, 
        prices: pd.Series, 
        periods: List[int] = None
    ) -> List[pd.Series]:
        """
        Calculate price momentum over various periods.
        
        Args:
            prices: Price series
            periods: List of momentum periods
            
        Returns:
            List of momentum values for each period
        """
        if periods is None:
            periods = [5, 10, 20]
        
        momentum_features = []
        
        try:
            for period in periods:
                mom = (prices / prices.shift(period) - 1).fillna(0)
                momentum_features.append(mom)
            
            return momentum_features
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return [pd.Series(0, index=prices.index) for _ in periods]
