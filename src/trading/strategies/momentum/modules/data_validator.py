"""
Momentum Strategy - Data Validator Module
Handles data validation and quality checking for the strategy
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class DataValidator:
    """Validate and monitor market data quality for momentum strategy"""
    
    def __init__(self, params: Dict):
        """
        Initialize data validator
        
        Args:
            params: Strategy parameters
        """
        self.params = params
        self.data_quality_issues = {}
        self.max_quality_issues = 5
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'quality_issues': 0
        }
    
    def validate_market_data(self, df: pd.DataFrame) -> bool:
        """Validate market data integrity"""
        self.validation_stats['total_validations'] += 1
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.error("Empty DataFrame provided")
                self.validation_stats['failed_validations'] += 1
                return False
            
            # Check minimum data points
            if len(df) < self.params['min_data_points']:
                logger.error(f"Insufficient data: {len(df)} rows, need {self.params['min_data_points']}")
                self.validation_stats['failed_validations'] += 1
                return False
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                self.validation_stats['failed_validations'] += 1
                return False
            
            # Validate price data
            if not self._validate_price_data(df):
                self.validation_stats['failed_validations'] += 1
                return False
            
            # Validate OHLC relationships
            if not self._validate_ohlc_relationships(df):
                self.validation_stats['failed_validations'] += 1
                return False
            
            # Check for extreme price movements
            if not self._check_price_movements(df):
                self.validation_stats['failed_validations'] += 1
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating market data: {e}")
            self.validation_stats['failed_validations'] += 1
            return False
    
    def _validate_price_data(self, df: pd.DataFrame) -> bool:
        """Validate price columns"""
        try:
            for col in ['open', 'high', 'low', 'close']:
                if df[col].isnull().any():
                    logger.error(f"Null values in {col}")
                    return False
                
                if (df[col] <= 0).any():
                    logger.error(f"Non-positive values in {col}")
                    return False
                
                # Check for infinite values
                if np.isinf(df[col]).any():
                    logger.error(f"Infinite values in {col}")
                    return False
            
            # Validate volume
            if df['volume'].isnull().any():
                logger.warning("Null values in volume, will be filled with 0")
                df['volume'].fillna(0, inplace=True)
            
            if (df['volume'] < 0).any():
                logger.error("Negative volume values")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating price data: {e}")
            return False
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> bool:
        """Validate OHLC candle relationships"""
        try:
            invalid_candles = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            
            if invalid_candles > 0:
                logger.error(f"Found {invalid_candles} invalid OHLC candles")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating OHLC relationships: {e}")
            return False
    
    def _check_price_movements(self, df: pd.DataFrame) -> bool:
        """Check for extreme price movements"""
        try:
            # Check for extreme price movements (>50% in one candle)
            price_changes = df['close'].pct_change().abs()
            extreme_moves = (price_changes > 0.5).sum()
            
            if extreme_moves > len(df) * 0.01:  # More than 1% extreme moves
                logger.warning(f"Detected {extreme_moves} extreme price movements")
                return False
            
            # Check for price spikes
            high_low_ratio = df['high'] / df['low']
            spikes = (high_low_ratio > 1.5).sum()  # 50% difference between high and low
            
            if spikes > len(df) * 0.05:  # More than 5% spikes
                logger.warning(f"Detected {spikes} price spikes")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking price movements: {e}")
            return False
    
    def has_quality_issues(self, symbol: str) -> bool:
        """Check if symbol has too many quality issues"""
        return self.data_quality_issues.get(symbol, 0) >= self.max_quality_issues
    
    def record_quality_issue(self, symbol: str):
        """Record data quality issue"""
        self.data_quality_issues[symbol] = self.data_quality_issues.get(symbol, 0) + 1
        self.validation_stats['quality_issues'] += 1
        logger.warning(f"Data quality issue recorded for {symbol}: count={self.data_quality_issues[symbol]}")
    
    def reset_quality_issues(self, symbol: str):
        """Reset quality issues for a symbol"""
        if symbol in self.data_quality_issues:
            del self.data_quality_issues[symbol]
            logger.info(f"Reset quality issues for {symbol}")
    
    def validate_parameters(self, params: Dict) -> bool:
        """Validate strategy parameters"""
        try:
            # RSI validation
            assert 0 < params['rsi_period'] < 100, "Invalid RSI period"
            assert 0 < params['rsi_oversold'] < params['rsi_overbought'] < 100, \
                "Invalid RSI thresholds"
            
            # MACD validation
            assert 0 < params['macd_fast'] < params['macd_slow'], \
                "Invalid MACD periods"
            assert 0 < params['macd_signal'] < 50, "Invalid MACD signal period"
            
            # ADX validation
            assert 0 < params['adx_period'] < 100, "Invalid ADX period"
            assert 0 < params['adx_threshold'] < 100, "Invalid ADX threshold"
            
            # Risk validation
            assert 0 < params['atr_multiplier_sl'] < 10, "Invalid SL multiplier"
            assert 0 < params['atr_multiplier_tp'] < 10, "Invalid TP multiplier"
            assert 0 < params['min_confidence'] <= 1, "Invalid confidence threshold"
            
            # Position limits
            assert params['max_positions'] > 0, "Invalid max positions"
            assert params['min_data_points'] > 0, "Invalid min data points"
            
            return True
            
        except AssertionError as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        stats['symbols_with_issues'] = len(self.data_quality_issues)
        stats['total_issues'] = sum(self.data_quality_issues.values())
        return stats
    
    def clean_old_issues(self, days: int = 7):
        """Clean old quality issues (simplified without timestamps)"""
        # In a real implementation, would track timestamps
        # For now, just clear if too many symbols
        if len(self.data_quality_issues) > 100:
            logger.info("Clearing old quality issues")
            self.data_quality_issues.clear()
