"""Market regime feature engineering module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import talib
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class RegimeFeatureEngineer:
    """Handles market regime feature creation"""
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime features"""
        df = df.copy()
        
        # Volatility regimes
        volatility = df['returns'].rolling(20).std()
        volatility_percentiles = volatility.rolling(252).rank(pct=True)
        
        df['volatility_regime'] = pd.cut(volatility_percentiles, 
                                         bins=[0, 0.25, 0.5, 0.75, 1.0],
                                         labels=[0, 1, 2, 3])
        
        # Trend regimes
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        df['trend_strength'] = (sma_20 - sma_50) / sma_50
        df['trend_regime'] = np.where(df['trend_strength'] > 0.02, 1,
                                     np.where(df['trend_strength'] < -0.02, -1, 0))
        
        # Volume regimes
        volume_ma = df['volume'].rolling(20).mean()
        df['volume_regime'] = np.where(df['volume'] > 1.5 * volume_ma, 1, 0)
        
        # Momentum regimes
        rsi = talib.RSI(df['close'], timeperiod=14)
        df['momentum_regime'] = np.where(rsi > 70, 1, np.where(rsi < 30, -1, 0))
        
        logger.debug("Created regime features")
        
        return df
    
    def create_fourier_features(self, df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """Create Fourier transform features"""
        df = df.copy()
        
        # Apply FFT to price series
        price_fft = np.fft.fft(df['close'].fillna(method='ffill'))
        freqs = np.fft.fftfreq(len(price_fft))
        
        # Get dominant frequencies
        power_spectrum = np.abs(price_fft) ** 2
        dominant_freq_indices = np.argsort(power_spectrum)[-n_components:]
        
        # Create features from dominant frequencies
        for i, idx in enumerate(dominant_freq_indices):
            df[f'fft_freq_{i}'] = freqs[idx]
            df[f'fft_power_{i}'] = power_spectrum[idx]
            
            # Reconstruct signal from this frequency
            single_freq_fft = np.zeros_like(price_fft)
            single_freq_fft[idx] = price_fft[idx]
            single_freq_fft[-idx] = price_fft[-idx]  # Symmetric component
            
            reconstructed = np.real(np.fft.ifft(single_freq_fft))
            df[f'fft_component_{i}'] = reconstructed
        
        logger.debug(f"Created {n_components} Fourier features")
        
        return df
