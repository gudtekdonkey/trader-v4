"""Market microstructure feature engineering module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class MicrostructureFeatureEngineer:
    """Handles market microstructure feature creation"""
    
    def create_microstructure_features(self, df: pd.DataFrame, orderbook_data: Dict) -> pd.DataFrame:
        """Create advanced market microstructure features"""
        df = df.copy()
        
        # Order flow imbalance
        bid_volume = orderbook_data.get('bid_volume', 0)
        ask_volume = orderbook_data.get('ask_volume', 0)
        
        if bid_volume + ask_volume > 0:
            df['order_flow_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            df['order_flow_imbalance'] = 0
        
        # Volume-weighted mid price
        bid_price = orderbook_data.get('bid_price', df['close'].iloc[-1])
        ask_price = orderbook_data.get('ask_price', df['close'].iloc[-1])
        
        if bid_volume + ask_volume > 0:
            df['volume_weighted_mid'] = (bid_price * ask_volume + ask_price * bid_volume) / (bid_volume + ask_volume)
        else:
            df['volume_weighted_mid'] = (bid_price + ask_price) / 2
        
        # Bid-ask spread metrics
        spread = orderbook_data.get('spread', ask_price - bid_price)
        df['bid_ask_spread'] = spread
        df['relative_spread'] = spread / ((bid_price + ask_price) / 2) if (bid_price + ask_price) > 0 else 0
        
        # Time-weighted spread
        df['time_weighted_spread'] = df['bid_ask_spread'].ewm(span=20).mean()
        
        # Pressure ratios
        if ask_volume > 0:
            df['pressure_ratio'] = bid_volume / ask_volume
        else:
            df['pressure_ratio'] = 1.0
        
        # Depth imbalance
        total_depth = bid_volume + ask_volume
        if total_depth > 0:
            df['depth_imbalance'] = (bid_volume - ask_volume) / total_depth
        else:
            df['depth_imbalance'] = 0
        
        # Price impact estimation
        if total_depth > 0:
            df['estimated_price_impact'] = df['volume'] / total_depth
        else:
            df['estimated_price_impact'] = 0
        
        # Rolling microstructure features
        for window in [5, 10, 20]:
            df[f'avg_spread_{window}'] = df['bid_ask_spread'].rolling(window).mean()
            df[f'spread_volatility_{window}'] = df['bid_ask_spread'].rolling(window).std()
            df[f'avg_imbalance_{window}'] = df['order_flow_imbalance'].rolling(window).mean()
        
        logger.debug("Created microstructure features")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between indicators"""
        df = df.copy()
        
        # Price vs Moving Averages
        if 'sma_20' in df.columns:
            df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
            df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        
        if 'sma_50' in df.columns:
            df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
            df['sma20_vs_sma50'] = df['sma_20'] / df['sma_50'] - 1
        
        # RSI interactions
        if 'rsi_14' in df.columns:
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_momentum'] = df['rsi_14'] - df['rsi_14'].shift(5)
        
        # Volume interactions
        df['volume_price_trend'] = df['volume'] * df['returns']
        df['high_volume_up'] = ((df['volume'] > df['volume'].rolling(20).mean()) & 
                               (df['returns'] > 0)).astype(int)
        df['high_volume_down'] = ((df['volume'] > df['volume'].rolling(20).mean()) & 
                                 (df['returns'] < 0)).astype(int)
        
        logger.debug("Created interaction features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str] = ['returns', 'volume'], 
                           lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features"""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Autoregressive features
        df['return_autocorr_10'] = df['returns'].rolling(10).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0
        )
        
        logger.debug(f"Created lag features for columns: {columns}")
        
        return df
