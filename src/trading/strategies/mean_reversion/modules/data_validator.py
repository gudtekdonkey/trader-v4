"""
Mean Reversion Strategy - Data Validator Module
Handles data validation and cleaning for the strategy
"""

import numpy as np
import pandas as pd
from typing import Optional
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class DataValidator:
    """Validate and clean market data for mean reversion strategy"""
    
    def __init__(self):
        """Initialize data validator"""
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'data_cleaned': 0
        }
    
    def validate_and_clean(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Validate and clean market data
        
        Args:
            df: Raw market data DataFrame
            
        Returns:
            Cleaned DataFrame or None if data is invalid
        """
        self.validation_stats['total_validations'] += 1
        
        try:
            # Check if DataFrame is empty
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for validation")
                self.validation_stats['failed_validations'] += 1
                return None
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                self.validation_stats['failed_validations'] += 1
                return None
            
            # Clean the data
            df_cleaned = self._clean_data(df.copy())
            
            # Validate cleaned data
            if df_cleaned.empty:
                logger.warning("Data cleaning resulted in empty DataFrame")
                self.validation_stats['failed_validations'] += 1
                return None
            
            # Check if we have enough data points
            if len(df_cleaned) < 20:  # Minimum for basic indicators
                logger.warning(f"Insufficient data after cleaning: {len(df_cleaned)} rows")
                self.validation_stats['failed_validations'] += 1
                return None
            
            self.validation_stats['data_cleaned'] += 1
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            self.validation_stats['failed_validations'] += 1
            return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare market data"""
        try:
            # Remove rows with invalid prices (zero or negative)
            initial_len = len(df)
            df = df[(df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['open'] > 0)]
            
            if len(df) < initial_len:
                logger.info(f"Removed {initial_len - len(df)} rows with invalid prices")
            
            # Handle NaN values
            if df.isnull().any().any():
                # First try forward fill, then backward fill
                df = df.fillna(method='ffill').fillna(method='bfill')
                logger.info("Filled NaN values in data")
            
            # Remove extreme outliers (prices > 10x median)
            median_price = df['close'].median()
            outlier_threshold = median_price * 10
            outliers = df['close'] > outlier_threshold
            
            if outliers.any():
                df = df[~outliers]
                logger.info(f"Removed {outliers.sum()} price outliers")
            
            # Ensure volume is non-negative
            negative_volume = df['volume'] < 0
            if negative_volume.any():
                df.loc[negative_volume, 'volume'] = 0
                logger.info(f"Fixed {negative_volume.sum()} negative volume values")
            
            # Validate OHLC relationships
            invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['close']) | (df['low'] > df['close'])
            if invalid_ohlc.any():
                # Fix by adjusting high/low
                df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, ['open', 'close', 'high']].max(axis=1)
                df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, ['open', 'close', 'low']].min(axis=1)
                logger.info(f"Fixed {invalid_ohlc.sum()} invalid OHLC relationships")
            
            # Sort by index if it's a time index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return pd.DataFrame()
    
    def validate_parameters(self, params: Dict) -> bool:
        """Validate strategy parameters"""
        try:
            # Check all required parameters exist
            required_params = [
                'lookback_period', 'entry_z_score', 'exit_z_score', 'max_z_score',
                'bb_period', 'bb_std', 'rsi_period', 'rsi_oversold', 'rsi_overbought',
                'min_volatility', 'max_volatility', 'atr_multiplier_sl',
                'position_hold_bars', 'min_confidence'
            ]
            
            for param in required_params:
                if param not in params:
                    logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate parameter ranges
            validations = [
                (params['lookback_period'] > 0, "Lookback period must be positive"),
                (params['entry_z_score'] > 0, "Entry z-score must be positive"),
                (params['exit_z_score'] >= 0, "Exit z-score must be non-negative"),
                (params['exit_z_score'] < params['entry_z_score'], "Exit z-score must be less than entry"),
                (params['max_z_score'] > params['entry_z_score'], "Max z-score must be greater than entry"),
                (params['bb_period'] > 0, "Bollinger Band period must be positive"),
                (params['bb_std'] > 0, "Bollinger Band std must be positive"),
                (1 <= params['rsi_period'] <= 100, "RSI period out of range"),
                (0 < params['rsi_oversold'] < params['rsi_overbought'] < 100, "Invalid RSI thresholds"),
                (0 < params['min_volatility'] < params['max_volatility'], "Invalid volatility range"),
                (params['atr_multiplier_sl'] > 0, "ATR multiplier must be positive"),
                (params['position_hold_bars'] > 0, "Position hold bars must be positive"),
                (0 <= params['min_confidence'] <= 1, "Confidence must be between 0 and 1")
            ]
            
            for condition, error_msg in validations:
                if not condition:
                    logger.error(f"Parameter validation failed: {error_msg}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating parameters: {e}")
            return False
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics"""
        return self.validation_stats.copy()
