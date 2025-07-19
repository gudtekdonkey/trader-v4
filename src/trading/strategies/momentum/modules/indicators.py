"""
Momentum Strategy - Indicators Module
Handles all technical indicator calculations for momentum signals
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Optional
import traceback
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class MomentumIndicators:
    """Calculate and manage momentum technical indicators"""
    
    def __init__(self, params: Dict):
        """
        Initialize indicator calculator with strategy parameters
        
        Args:
            params: Strategy parameters dictionary
        """
        self.params = params
        self.calculation_cache = {}
        self.cache_expiry = 60  # seconds
        self.calculation_failures = 0
    
    def calculate_all(self, df: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
        """Calculate all momentum indicators with comprehensive error handling"""
        indicators = {}
        
        try:
            # Price momentum with error handling
            indicators['rsi'] = self._calculate_rsi(df['close'])
            
            # MACD indicators
            macd_data = self._calculate_macd(df['close'])
            indicators.update(macd_data)
            
            # Trend strength indicators
            adx_data = self._calculate_adx(df)
            indicators.update(adx_data)
            
            # Volume analysis
            volume_data = self._calculate_volume_metrics(df)
            indicators.update(volume_data)
            
            # Volatility indicator
            indicators['atr'] = self._calculate_atr(df)
            
            # Price position
            indicators['close_position'] = self._calculate_price_position(df)
            
            # Rate of change
            roc_data = self._calculate_roc(df['close'])
            indicators.update(roc_data)
            
            # Moving averages
            ma_data = self._calculate_moving_averages(df['close'])
            indicators.update(ma_data)
            
            # Validate and clean all indicators
            indicators = self._validate_indicators(indicators, df)
            
            self.calculation_failures = 0  # Reset on success
            return indicators
            
        except Exception as e:
            logger.error(f"Critical error calculating indicators: {e}")
            logger.error(traceback.format_exc())
            self.calculation_failures += 1
            return None
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI with error handling"""
        try:
            rsi = talib.RSI(prices.values, timeperiod=self.params['rsi_period'])
            return pd.Series(rsi, index=prices.index)
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return self._calculate_rsi_fallback(prices, self.params['rsi_period'])
    
    def _calculate_rsi_fallback(self, prices: pd.Series, period: int) -> pd.Series:
        """Simple RSI calculation fallback"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Neutral RSI for NaN
        except Exception:
            return pd.Series(np.full(len(prices), 50), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicators with error handling"""
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                prices.values,
                fastperiod=self.params['macd_fast'],
                slowperiod=self.params['macd_slow'],
                signalperiod=self.params['macd_signal']
            )
            return {
                'macd': pd.Series(macd, index=prices.index),
                'macd_signal': pd.Series(macd_signal, index=prices.index),
                'macd_hist': pd.Series(macd_hist, index=prices.index)
            }
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            # Return neutral values
            return {
                'macd': pd.Series(np.zeros(len(prices)), index=prices.index),
                'macd_signal': pd.Series(np.zeros(len(prices)), index=prices.index),
                'macd_hist': pd.Series(np.zeros(len(prices)), index=prices.index)
            }
    
    def _calculate_adx(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate ADX and directional indicators"""
        try:
            adx = talib.ADX(
                df['high'].values, 
                df['low'].values, 
                df['close'].values, 
                timeperiod=self.params['adx_period']
            )
            plus_di = talib.PLUS_DI(
                df['high'].values, 
                df['low'].values, 
                df['close'].values, 
                timeperiod=self.params['adx_period']
            )
            minus_di = talib.MINUS_DI(
                df['high'].values, 
                df['low'].values, 
                df['close'].values, 
                timeperiod=self.params['adx_period']
            )
            return {
                'adx': pd.Series(adx, index=df.index),
                'plus_di': pd.Series(plus_di, index=df.index),
                'minus_di': pd.Series(minus_di, index=df.index)
            }
        except Exception as e:
            logger.error(f"ADX calculation failed: {e}")
            # Return neutral values
            return {
                'adx': pd.Series(np.full(len(df), 25), index=df.index),
                'plus_di': pd.Series(np.full(len(df), 50), index=df.index),
                'minus_di': pd.Series(np.full(len(df), 50), index=df.index)
            }
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based metrics"""
        try:
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'] / volume_sma
            # Replace inf/nan with 1
            volume_ratio = volume_ratio.replace([np.inf, -np.inf], 1).fillna(1)
            
            return {
                'volume_sma': volume_sma,
                'volume_ratio': volume_ratio
            }
        except Exception as e:
            logger.error(f"Volume calculation failed: {e}")
            return {
                'volume_sma': df['volume'],
                'volume_ratio': pd.Series(np.ones(len(df)), index=df.index)
            }
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        try:
            atr = talib.ATR(
                df['high'].values, 
                df['low'].values, 
                df['close'].values, 
                timeperiod=14
            )
            return pd.Series(atr, index=df.index)
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return self._calculate_atr_fallback(df)
    
    def _calculate_atr_fallback(self, df: pd.DataFrame) -> pd.Series:
        """Simple ATR calculation fallback"""
        try:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            
            return atr.fillna(true_range.mean())
        except Exception:
            # Return 2% of price as default ATR
            return df['close'] * 0.02
    
    def _calculate_price_position(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price position within rolling range"""
        try:
            rolling_min = df['low'].rolling(20).min()
            rolling_max = df['high'].rolling(20).max()
            price_range = rolling_max - rolling_min
            
            # Avoid division by zero
            close_position = pd.Series(np.where(
                price_range > 0,
                (df['close'] - rolling_min) / price_range,
                0.5
            ), index=df.index)
            
            return close_position
        except Exception as e:
            logger.error(f"Price position calculation failed: {e}")
            return pd.Series(np.full(len(df), 0.5), index=df.index)
    
    def _calculate_roc(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Rate of Change for multiple periods"""
        roc_data = {}
        
        for period in [5, 10, 20]:
            try:
                roc = talib.ROC(prices.values, timeperiod=period)
                # Replace extreme values
                roc = pd.Series(roc, index=prices.index)
                roc = roc.replace([np.inf, -np.inf], 0)
                roc = roc.clip(-50, 50)  # Cap at ±50%
                roc_data[f'roc_{period}'] = roc
            except Exception as e:
                logger.error(f"ROC{period} calculation failed: {e}")
                roc_data[f'roc_{period}'] = pd.Series(np.zeros(len(prices)), index=prices.index)
        
        return roc_data
    
    def _calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate various moving averages"""
        ma_data = {}
        ma_periods = {'sma_20': 20, 'sma_50': 50, 'ema_9': 9, 'ema_21': 21}
        
        for name, period in ma_periods.items():
            try:
                if 'sma' in name:
                    ma_data[name] = pd.Series(
                        talib.SMA(prices.values, timeperiod=period),
                        index=prices.index
                    )
                else:
                    ma_data[name] = pd.Series(
                        talib.EMA(prices.values, timeperiod=period),
                        index=prices.index
                    )
            except Exception as e:
                logger.error(f"{name} calculation failed: {e}")
                # Simple moving average fallback
                ma_data[name] = prices.rolling(period).mean()
        
        return ma_data
    
    def _validate_indicators(self, indicators: Dict[str, pd.Series], df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Validate and clean all indicators"""
        validated = {}
        
        for key, value in indicators.items():
            # Convert to pandas Series if needed
            if not isinstance(value, pd.Series):
                value = pd.Series(value, index=df.index)
            
            # Replace any remaining NaN/inf
            value = value.replace([np.inf, -np.inf], np.nan)
            value = value.fillna(method='ffill').fillna(method='bfill')
            
            # Final fallback for any remaining NaN
            if key == 'rsi':
                value = value.fillna(50)
            elif key in ['adx', 'plus_di', 'minus_di']:
                value = value.fillna(25)
            elif key in ['volume_ratio']:
                value = value.fillna(1)
            else:
                value = value.fillna(0)
            
            validated[key] = value
        
        return validated
    
    def clean_cache(self):
        """Clean expired cache entries"""
        current_time = pd.Timestamp.now()
        expired_keys = [
            key for key in self.calculation_cache
            if (current_time - pd.Timestamp(key.split('_')[-1])).seconds > self.cache_expiry
        ]
        for key in expired_keys:
            del self.calculation_cache[key]
    
    def get_calculation_failures(self) -> int:
        """Get number of calculation failures"""
        return self.calculation_failures
