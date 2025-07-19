"""
Mean Reversion Strategy - Indicators Module
Handles all technical indicator calculations for mean reversion signals
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Optional
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class MeanReversionIndicators:
    """Calculate and manage mean reversion technical indicators"""
    
    def __init__(self, params: Dict):
        """
        Initialize indicator calculator with strategy parameters
        
        Args:
            params: Strategy parameters dictionary
        """
        self.params = params
    
    def calculate_all(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all mean reversion indicators with comprehensive error handling"""
        indicators = {}
        
        try:
            # Price statistics with validation
            indicators.update(self._calculate_price_statistics(df))
            
            # Bollinger Bands with validation
            indicators.update(self._calculate_bollinger_bands(df))
            
            # RSI calculation with validation
            indicators['rsi'] = self._calculate_rsi(df['close'])
            
            # Volatility with validation
            indicators['volatility'] = self._calculate_volatility(df)
            
            # ATR calculation with validation
            indicators['atr'] = self._calculate_atr(df)
            
            # Volume analysis with validation
            indicators.update(self._calculate_volume_metrics(df))
            
            # Mean reversion speed with validation
            indicators['half_life'] = self._calculate_half_life(df['close'])
            
            # Hurst exponent with validation
            indicators['hurst_exponent'] = self._calculate_hurst_exponent(df['close'])
            
            # Price efficiency ratio with validation
            indicators['efficiency_ratio'] = self._calculate_efficiency_ratio(df['close'])
            
            return indicators
            
        except Exception as e:
            logger.error(f"Critical error calculating indicators: {e}")
            return {}
    
    def _calculate_price_statistics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate price-based statistics"""
        try:
            sma = df['close'].rolling(self.params['lookback_period']).mean()
            std = df['close'].rolling(self.params['lookback_period']).std()
            
            # Handle division by zero in z-score
            z_score = pd.Series(0, index=df.index)
            valid_std = std > 0
            z_score[valid_std] = (df['close'][valid_std] - sma[valid_std]) / std[valid_std]
            z_score = z_score.fillna(0)
            
            return {
                'sma': sma,
                'std': std,
                'z_score': z_score
            }
        except Exception as e:
            logger.error(f"Error calculating price statistics: {e}")
            # Fallback values
            return {
                'sma': df['close'],
                'std': pd.Series(df['close'].std(), index=df.index),
                'z_score': pd.Series(0, index=df.index)
            }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands with error handling"""
        try:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'],
                timeperiod=self.params['bb_period'],
                nbdevup=self.params['bb_std'],
                nbdevdn=self.params['bb_std']
            )
            
            # Calculate BB position safely
            bb_range = bb_upper - bb_lower
            bb_position = pd.Series(0.5, index=df.index)
            valid_range = bb_range > 0
            bb_position[valid_range] = (
                (df['close'][valid_range] - bb_lower[valid_range]) / bb_range[valid_range]
            )
            bb_position = bb_position.clip(0, 1)
            
            return {
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_position': bb_position
            }
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}, using fallback")
            # Fallback BB calculation
            sma = df['close'].rolling(self.params['bb_period']).mean()
            std = df['close'].rolling(self.params['bb_period']).std()
            return {
                'bb_upper': sma + (std * self.params['bb_std']),
                'bb_middle': sma,
                'bb_lower': sma - (std * self.params['bb_std']),
                'bb_position': pd.Series(0.5, index=df.index)
            }
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI with fallback"""
        try:
            rsi = talib.RSI(prices, timeperiod=self.params['rsi_period'])
            return rsi.fillna(50)
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}, using fallback")
            return self._calculate_rsi_fallback(prices, self.params['rsi_period'])
    
    def _calculate_rsi_fallback(self, prices: pd.Series, period: int) -> pd.Series:
        """Fallback RSI calculation"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Neutral value for NaN
        except Exception as e:
            logger.error(f"RSI fallback calculation failed: {e}")
            return pd.Series(50, index=prices.index)
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price volatility"""
        try:
            if 'returns' in df:
                volatility = df['returns'].rolling(20).std()
            else:
                volatility = df['close'].pct_change().rolling(20).std()
            return volatility.fillna(0.02)  # Default 2% volatility
        except Exception as e:
            logger.warning(f"Volatility calculation failed: {e}")
            return pd.Series(0.02, index=df.index)
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        try:
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            return atr.fillna(df['close'] * 0.02)  # 2% of price as default
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}, using fallback")
            return self._calculate_atr_fallback(df, 14)
    
    def _calculate_atr_fallback(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Fallback ATR calculation"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            # Fill NaN with percentage of price
            return atr.fillna(close * 0.02)  # 2% of price as default
        except Exception as e:
            logger.error(f"ATR fallback calculation failed: {e}")
            return df['close'] * 0.02  # 2% of close price
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based metrics"""
        try:
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'] / volume_sma
            # Handle division by zero and inf values
            volume_ratio = volume_ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0)
            
            return {
                'volume_sma': volume_sma,
                'volume_ratio': volume_ratio
            }
        except Exception as e:
            logger.warning(f"Volume analysis failed: {e}")
            return {
                'volume_sma': df['volume'],
                'volume_ratio': pd.Series(1.0, index=df.index)
            }
    
    def _calculate_half_life(self, prices: pd.Series, window: int = 60) -> pd.Series:
        """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process"""
        half_lives = pd.Series(index=prices.index, dtype=float)
        
        try:
            for i in range(window, len(prices)):
                try:
                    y = prices.iloc[i-window:i].values
                    y_lag = prices.iloc[i-window-1:i-1].values
                    
                    # Run regression: y_t = alpha + beta * y_{t-1} + epsilon
                    X = np.column_stack([np.ones(len(y_lag)), y_lag])
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                    # Half-life = -log(2) / log(beta[1])
                    if 0 < beta[1] < 1:
                        half_lives.iloc[i] = -np.log(2) / np.log(beta[1])
                    else:
                        half_lives.iloc[i] = np.nan
                except Exception:
                    half_lives.iloc[i] = np.nan
            
            return half_lives.fillna(method='ffill').fillna(24)  # Default 24 periods
        except Exception as e:
            logger.error(f"Error in half-life calculation: {e}")
            return pd.Series(24, index=prices.index)
    
    def _calculate_hurst_exponent(self, prices: pd.Series, window: int = 100) -> pd.Series:
        """Calculate Hurst exponent to measure mean reversion tendency"""
        hurst_values = pd.Series(index=prices.index, dtype=float)
        
        try:
            for i in range(window, len(prices)):
                try:
                    data = prices.iloc[i-window:i].values
                    
                    # Calculate R/S statistic
                    mean_data = np.mean(data)
                    deviations = data - mean_data
                    cumulative_deviations = np.cumsum(deviations)
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    S = np.std(data, ddof=1)
                    
                    if S != 0:
                        RS = R / S
                        # Hurst exponent approximation
                        hurst_values.iloc[i] = np.log(RS) / np.log(window)
                    else:
                        hurst_values.iloc[i] = 0.5
                except Exception:
                    hurst_values.iloc[i] = 0.5
            
            return hurst_values.fillna(0.5)
        except Exception as e:
            logger.error(f"Error in Hurst exponent calculation: {e}")
            return pd.Series(0.5, index=prices.index)
    
    def _calculate_efficiency_ratio(self, prices: pd.Series) -> pd.Series:
        """Calculate price efficiency ratio"""
        try:
            change = (prices - prices.shift(self.params['lookback_period'])).abs()
            path = prices.diff().abs().rolling(self.params['lookback_period']).sum()
            efficiency_ratio = change / (path + 1e-10)
            efficiency_ratio = efficiency_ratio.fillna(0.5).clip(0, 1)
            return efficiency_ratio
        except Exception as e:
            logger.warning(f"Efficiency ratio calculation failed: {e}")
            return pd.Series(0.5, index=prices.index)
