"""Wavelet decomposition feature engineering module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pywt
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class WaveletFeatureEngineer:
    """Handles wavelet decomposition features"""
    
    def __init__(self):
        self.wavelet = 'db4'  # Daubechies wavelet
        self.decomposition_level = 4
    
    def create_wavelet_features(self, df: pd.DataFrame, columns: List[str] = ['close', 'volume']) -> pd.DataFrame:
        """Create wavelet decomposition features"""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(df[col].fillna(method='ffill'), self.wavelet, level=self.decomposition_level)
                
                # Reconstruct signals at different frequencies
                for i in range(len(coeffs)):
                    if i == 0:
                        # Approximation coefficient
                        reconstructed = pywt.waverec([coeffs[0]] + [None] * (len(coeffs) - 1), self.wavelet)
                        df[f'{col}_wavelet_trend'] = self._match_length(reconstructed, len(df))
                    else:
                        # Detail coefficients
                        detail_coeffs = [None] * len(coeffs)
                        detail_coeffs[i] = coeffs[i]
                        reconstructed = pywt.waverec(detail_coeffs, self.wavelet)
                        df[f'{col}_wavelet_d{i}'] = self._match_length(reconstructed, len(df))
                
                # Wavelet energy
                for i, coeff in enumerate(coeffs):
                    energy = np.sum(coeff ** 2)
                    df[f'{col}_wavelet_energy_{i}'] = energy
        
        logger.debug(f"Created wavelet features for columns: {columns}")
        
        return df
    
    def _match_length(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Match signal length to target length"""
        if len(signal) > target_length:
            return signal[:target_length]
        elif len(signal) < target_length:
            return np.pad(signal, (0, target_length - len(signal)), 'edge')
        return signal
