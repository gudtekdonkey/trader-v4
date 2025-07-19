"""Time-based feature engineering module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class TimeFeatureEngineer:
    """Handles creation of time-based features"""
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime index"""
        df = df.copy()
        
        # Hour of day (for 24/7 crypto markets)
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['dayofweek'] = df.index.dayofweek
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Day of month
        df['day'] = df.index.day
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)
        
        # Month
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Is weekend
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        logger.debug(f"Created {len([col for col in df.columns if any(x in col for x in ['hour', 'day', 'month', 'quarter', 'weekend'])])} time features")
        
        return df
