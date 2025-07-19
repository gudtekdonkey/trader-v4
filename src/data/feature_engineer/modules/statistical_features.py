"""Statistical feature engineering module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class StatisticalFeatureEngineer:
    """Handles statistical feature creation"""
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        df = df.copy()
        
        # Returns statistics
        for window in [5, 10, 20, 50]:
            # Moments
            df[f'return_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'return_std_{window}'] = df['returns'].rolling(window).std()
            df[f'return_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'return_kurt_{window}'] = df['returns'].rolling(window).kurt()
            
            # Percentiles
            df[f'return_25pct_{window}'] = df['returns'].rolling(window).quantile(0.25)
            df[f'return_75pct_{window}'] = df['returns'].rolling(window).quantile(0.75)
            
            # Range statistics
            df[f'high_low_ratio_{window}'] = df['high'].rolling(window).max() / df['low'].rolling(window).min()
            df[f'close_range_pct_{window}'] = (df['close'] - df['low'].rolling(window).min()) / \
                                              (df['high'].rolling(window).max() - df['low'].rolling(window).min())
        
        # Entropy
        df['return_entropy_20'] = df['returns'].rolling(20).apply(
            lambda x: stats.entropy(np.histogram(x, bins=10)[0] + 1e-10)
        )
        
        logger.debug("Created statistical features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced rolling window features"""
        df = df.copy()
        
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Price features
            df[f'min_max_ratio_{window}'] = df['low'].rolling(window).min() / df['high'].rolling(window).max()
            df[f'close_position_{window}'] = (df['close'] - df['low'].rolling(window).min()) / \
                                            (df['high'].rolling(window).max() - df['low'].rolling(window).min())
            
            # Volume features
            df[f'volume_trend_{window}'] = df['volume'].rolling(window).mean() / \
                                          df['volume'].rolling(window * 2).mean()
            
            # Efficiency ratio (Kaufman)
            change = (df['close'] - df['close'].shift(window)).abs()
            volatility = df['close'].diff().abs().rolling(window).sum()
            df[f'efficiency_ratio_{window}'] = change / (volatility + 1e-10)
            
            # Number of positive returns
            df[f'positive_returns_{window}'] = (df['returns'] > 0).rolling(window).sum() / window
        
        logger.debug("Created rolling window features")
        
        return df
