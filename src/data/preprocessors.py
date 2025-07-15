import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import talib
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """Advanced data preprocessing for ML models"""
    
    def __init__(self):
        self.scalers = {
            'price': RobustScaler(),
            'volume': RobustScaler(),
            'technical': StandardScaler(),
            'microstructure': StandardScaler()
        }
        
        self.pca_components = {}
        self.feature_stats = {}
        
    def prepare_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare OHLCV data with additional features"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price relative to OHLC
        df['close_to_open'] = (df['close'] - df['open']) / df['open']
        df['high_to_low'] = (df['high'] - df['low']) / df['low']
        df['close_to_high'] = (df['close'] - df['high']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['low']
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # Volatility features
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_60'] = df['returns'].rolling(60).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_60']
        
        # Price position
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / \
                                  (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        df['price_position_60'] = (df['close'] - df['close'].rolling(60).min()) / \
                                  (df['close'].rolling(60).max() - df['close'].rolling(60).min())
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = df.copy()
        
        # Trend indicators
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_10'] = talib.EMA(df['close'], timeperiod=10)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # RSI
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_28'] = talib.RSI(df['close'], timeperiod=28)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_28'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=28)
        
        # ADX (Average Directional Index)
        df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # MFI (Money Flow Index)
        df['mfi_14'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # OBV (On Balance Volume)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv_ema'] = talib.EMA(df['obv'], timeperiod=20)
        
        # Williams %R
        df['willr_14'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Parabolic SAR
        df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        df['sar_signal'] = np.where(df['close'] > df['sar'], 1, -1)
        
        # Pattern recognition
        df['cdl_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['cdl_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['cdl_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        
        return df
    
    def calculate_microstructure_features(self, df: pd.DataFrame, orderbook_data: Optional[Dict] = None) -> pd.DataFrame:
        """Calculate market microstructure features"""
        df = df.copy()
        
        if orderbook_data:
            # Order book imbalance
            df['order_imbalance'] = orderbook_data.get('imbalance', 0)
            df['bid_ask_spread'] = orderbook_data.get('spread', 0)
            df['mid_price'] = orderbook_data.get('mid_price', df['close'].iloc[-1])
            
            # Liquidity metrics
            df['bid_volume'] = orderbook_data.get('bid_volume', 0)
            df['ask_volume'] = orderbook_data.get('ask_volume', 0)
            df['total_liquidity'] = df['bid_volume'] + df['ask_volume']
        
        # Trade-based features
        df['trade_intensity'] = df['volume'] / df['volume'].rolling(20).mean()
        df['dollar_volume'] = df['close'] * df['volume']
        
        # Kyle's Lambda (price impact)
        if len(df) > 20:
            returns = df['returns'].dropna()
            volumes = df['volume'][returns.index]
            
            if len(returns) > 0 and volumes.std() > 0:
                df['kyle_lambda'] = returns.rolling(20).apply(
                    lambda x: np.abs(x).sum() / volumes.rolling(20).sum() if volumes.rolling(20).sum() > 0 else 0
                )
        
        # Amihud illiquidity
        df['amihud_illiquidity'] = np.abs(df['returns']) / df['dollar_volume']
        df['amihud_ratio'] = df['amihud_illiquidity'] / df['amihud_illiquidity'].rolling(20).mean()
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 60, 
                        prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        # Select features
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Remove NaN values
        df_clean = df.dropna()
        
        # Prepare data
        features = df_clean[feature_cols].values
        targets = df_clean['close'].shift(-prediction_horizon).values
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features) - prediction_horizon):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
