import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pywt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
import talib
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for crypto trading"""
    
    def __init__(self):
        self.wavelet = 'db4'  # Daubechies wavelet
        self.decomposition_level = 4
        self.pca_models = {}
        
    def engineer_all_features(self, df: pd.DataFrame, orderbook_data: Optional[Dict] = None) -> pd.DataFrame:
        """Engineer all features for the dataset"""
        logger.info("Starting comprehensive feature engineering")
        
        # Time-based features
        df = self.create_time_features(df)
        
        # Market microstructure features (NEW - HIGH IMPACT)
        if orderbook_data:
            df = self.create_microstructure_features(df, orderbook_data)
        
        # Wavelet features
        df = self.create_wavelet_features(df)
        
        # Statistical features
        df = self.create_statistical_features(df)
        
        # Fourier features
        df = self.create_fourier_features(df)
        
        # Market regime features
        df = self.create_regime_features(df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        # Lag features
        df = self.create_lag_features(df)
        
        # Rolling window features
        df = self.create_rolling_features(df)
        
        # Alternative data features (NEW)
        df = self.create_alternative_data_features(df)
        
        # Multi-timeframe features (NEW)
        df = self.create_multi_timeframe_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
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
        
        return df
    
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
        
        return df
    
    def _match_length(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Match signal length to target length"""
        if len(signal) > target_length:
            return signal[:target_length]
        elif len(signal) < target_length:
            return np.pad(signal, (0, target_length - len(signal)), 'edge')
        return signal
    
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
        
        return df
    
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
        
        return df
    
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
        
        return df
    
    def create_alternative_data_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create alternative data features (sentiment, macro, on-chain)"""
        df = df.copy()
        
        # Placeholder for real alternative data - implement with actual APIs
        # Social sentiment (would integrate with Twitter/Reddit APIs)
        df['social_sentiment'] = np.random.uniform(-1, 1, len(df))  # Placeholder
        df['social_volume'] = np.random.uniform(0, 1, len(df))  # Placeholder
        
        # Fear & Greed Index (would integrate with actual API)
        df['fear_greed_index'] = np.random.uniform(0, 100, len(df))  # Placeholder
        
        # On-chain metrics (would integrate with blockchain APIs)
        df['active_addresses'] = np.random.uniform(0.8, 1.2, len(df))  # Normalized placeholder
        df['whale_movements'] = np.random.uniform(-0.1, 0.1, len(df))  # Placeholder
        df['exchange_flows'] = np.random.uniform(-0.05, 0.05, len(df))  # Placeholder
        
        # Macro economic indicators (would integrate with economic APIs)
        df['vix_proxy'] = np.random.uniform(10, 30, len(df))  # Placeholder
        df['dxy_proxy'] = np.random.uniform(95, 105, len(df))  # Placeholder
        
        # Market structure metrics
        df['market_cap_dominance'] = np.random.uniform(0.4, 0.6, len(df))  # Placeholder
        
        # TODO: Replace placeholders with actual API integrations
        # Example implementations:
        # df['social_sentiment'] = self.get_social_sentiment_score()
        # df['fear_greed_index'] = self.get_fear_greed_index()
        # df['active_addresses'] = self.get_onchain_metrics()['active_addresses']
        
        return df
    
    def create_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multi-timeframe analysis features"""
        df = df.copy()
        
        # Simulate different timeframe data (in practice, would fetch actual multi-timeframe data)
        # Higher timeframe trend (4H equivalent)
        df['htf_trend'] = df['close'].rolling(240).mean() > df['close'].rolling(480).mean()  # 4H vs 8H
        df['htf_momentum'] = talib.RSI(df['close'], timeperiod=240)  # 4H RSI equivalent
        
        # Medium timeframe (1H equivalent)
        df['mtf_trend'] = df['close'].rolling(60).mean() > df['close'].rolling(120).mean()  # 1H vs 2H
        df['mtf_momentum'] = talib.RSI(df['close'], timeperiod=60)  # 1H RSI equivalent
        
        # Short timeframe (15M equivalent)
        df['stf_trend'] = df['close'].rolling(15).mean() > df['close'].rolling(30).mean()  # 15M vs 30M
        df['stf_momentum'] = talib.RSI(df['close'], timeperiod=15)  # 15M RSI equivalent
        
        # Timeframe agreement
        df['trend_agreement'] = (df['htf_trend'].astype(int) + 
                               df['mtf_trend'].astype(int) + 
                               df['stf_trend'].astype(int)) / 3
        
        # Momentum divergence
        df['momentum_divergence'] = abs(df['htf_momentum'] - df['stf_momentum'])
        
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
        
        return df
