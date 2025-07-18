"""
Feature Extractor Module

Handles feature extraction for regime detection including
price-based features, volume features, and technical indicators.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features for market regime detection.
    
    This class handles the extraction of various features from market data
    including price-based features, volume metrics, and technical indicators.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.required_columns = ['close']
        self.feature_names = []
    
    def extract_features(self, data: pd.DataFrame, technical_indicators) -> np.ndarray:
        """
        Extract features for regime detection with comprehensive error handling.
        
        Args:
            data: DataFrame with OHLCV and additional market data
            technical_indicators: Technical indicators calculator instance
            
        Returns:
            Feature matrix for regime detection
        """
        try:
            # Validate input
            if data is None or data.empty:
                logger.error("Empty data provided for feature extraction")
                return np.array([])
            
            # Check required columns
            if not all(col in data.columns for col in self.required_columns):
                logger.error(f"Missing required columns in data")
                return np.array([])
            
            features = []
            
            # Price-based features
            price_features = self._extract_price_features(data)
            features.extend(price_features)
            
            # Volume features
            volume_features = self._extract_volume_features(data)
            features.extend(volume_features)
            
            # Technical indicators
            tech_features = self._extract_technical_features(data, technical_indicators)
            features.extend(tech_features)
            
            # Market microstructure features
            micro_features = self._extract_microstructure_features(data)
            features.extend(micro_features)
            
            # Convert to numpy array
            return self._convert_to_array(features, data)
            
        except Exception as e:
            logger.error(f"Critical error in feature extraction: {e}")
            return np.array([])
    
    def _extract_price_features(self, data: pd.DataFrame) -> list:
        """Extract price-based features."""
        features = []
        
        try:
            returns = data['close'].pct_change()
            
            # Handle first return being NaN
            returns = returns.fillna(0)
            
            # Rolling statistics with min_periods
            features.append(returns.rolling(20, min_periods=1).mean())   # Trend
            features.append(returns.rolling(20, min_periods=1).std())    # Volatility
            features.append(returns.rolling(20, min_periods=1).skew())   # Skewness
            features.append(returns.rolling(20, min_periods=1).kurt())   # Kurtosis
            
            # Log returns
            log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
            features.append(log_returns.rolling(20, min_periods=1).mean())
            features.append(log_returns.rolling(20, min_periods=1).std())
            
            # Price momentum
            features.append((data['close'] / data['close'].shift(5) - 1).fillna(0))  # 5-period momentum
            features.append((data['close'] / data['close'].shift(20) - 1).fillna(0))  # 20-period momentum
            
        except Exception as e:
            logger.error(f"Error calculating price features: {e}")
            # Add placeholder features
            features.extend([pd.Series(0, index=data.index)] * 8)
        
        return features
    
    def _extract_volume_features(self, data: pd.DataFrame) -> list:
        """Extract volume-based features."""
        features = []
        
        try:
            if 'volume' in data.columns and data['volume'].notna().any():
                # Volume ratio
                volume_ratio = data['volume'] / data['volume'].rolling(20, min_periods=1).mean()
                volume_ratio = volume_ratio.replace([np.inf, -np.inf], 1).fillna(1)
                features.append(volume_ratio)
                
                # Volume volatility
                volume_returns = data['volume'].pct_change().fillna(0)
                features.append(volume_returns.rolling(20, min_periods=1).std())
                
                # On-balance volume indicator
                obv = self._calculate_obv(data)
                features.append(obv)
                
                # Volume-price correlation
                vp_corr = data['close'].rolling(20).corr(data['volume']).fillna(0)
                features.append(vp_corr)
            else:
                # No volume data, use placeholders
                features.extend([pd.Series(1, index=data.index)] * 4)
                
        except Exception as e:
            logger.error(f"Error calculating volume features: {e}")
            features.extend([pd.Series(1, index=data.index)] * 4)
        
        return features
    
    def _extract_technical_features(self, data: pd.DataFrame, technical_indicators) -> list:
        """Extract technical indicator features."""
        features = []
        
        try:
            # RSI
            features.append(technical_indicators.calculate_rsi(data['close'], 14))
            
            # ADX
            features.append(technical_indicators.calculate_adx(data, 14))
            
            # Bollinger Bands
            bb_features = technical_indicators.calculate_bollinger_bands(data['close'], 20, 2)
            features.extend(bb_features)
            
            # MACD
            macd_features = technical_indicators.calculate_macd(data['close'])
            features.extend(macd_features)
            
            # ATR (Average True Range)
            features.append(technical_indicators.calculate_atr(data, 14))
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Add placeholder features
            features.extend([pd.Series(50, index=data.index)] * 7)
        
        return features
    
    def _extract_microstructure_features(self, data: pd.DataFrame) -> list:
        """Extract market microstructure features."""
        features = []
        
        # Bid-ask spread
        if 'bid_ask_spread' in data.columns:
            try:
                spread_mean = data['bid_ask_spread'].rolling(20, min_periods=1).mean()
                features.append(spread_mean.fillna(0))
                
                spread_std = data['bid_ask_spread'].rolling(20, min_periods=1).std()
                features.append(spread_std.fillna(0))
            except Exception as e:
                logger.error(f"Error calculating spread features: {e}")
                features.extend([pd.Series(0, index=data.index)] * 2)
        else:
            features.extend([pd.Series(0, index=data.index)] * 2)
        
        # Order flow imbalance
        if 'order_flow_imbalance' in data.columns:
            try:
                ofi_mean = data['order_flow_imbalance'].rolling(20, min_periods=1).mean()
                features.append(ofi_mean.fillna(0))
                
                ofi_std = data['order_flow_imbalance'].rolling(20, min_periods=1).std()
                features.append(ofi_std.fillna(0))
            except Exception as e:
                logger.error(f"Error calculating order flow features: {e}")
                features.extend([pd.Series(0, index=data.index)] * 2)
        else:
            features.extend([pd.Series(0, index=data.index)] * 2)
        
        # High-low spread
        if 'high' in data.columns and 'low' in data.columns:
            try:
                hl_spread = (data['high'] - data['low']) / data['close']
                features.append(hl_spread.rolling(20, min_periods=1).mean().fillna(0))
            except Exception as e:
                logger.error(f"Error calculating high-low spread: {e}")
                features.append(pd.Series(0, index=data.index))
        else:
            features.append(pd.Series(0, index=data.index))
        
        return features
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume indicator."""
        try:
            obv = pd.Series(0, index=data.index)
            obv.iloc[0] = data['volume'].iloc[0]
            
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            # Normalize OBV
            obv_norm = (obv - obv.rolling(20, min_periods=1).mean()) / obv.rolling(20, min_periods=1).std()
            return obv_norm.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return pd.Series(0, index=data.index)
    
    def _convert_to_array(self, features: list, data: pd.DataFrame) -> np.ndarray:
        """Convert feature list to numpy array with error handling."""
        try:
            # Ensure all features have same length
            min_length = min(len(f) for f in features)
            features = [f.iloc[:min_length] for f in features]
            
            feature_matrix = np.column_stack([
                f.fillna(0).values for f in features
            ])
            
            # Replace any remaining NaN/Inf
            feature_matrix = np.nan_to_num(
                feature_matrix, 
                nan=0.0, 
                posinf=1e6, 
                neginf=-1e6
            )
            
            return feature_matrix
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            return np.zeros((len(data), len(features)))
