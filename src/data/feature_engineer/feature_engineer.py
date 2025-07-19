"""
Feature Engineering Coordinator - Advanced feature creation system
Orchestrates multiple feature engineering modules to create comprehensive
technical, statistical, wavelet, and microstructure features.

File: feature_engineer.py
Modified: 2025-07-18
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA

from .modules.time_features import TimeFeatureEngineer
from .modules.wavelet_features import WaveletFeatureEngineer
from .modules.statistical_features import StatisticalFeatureEngineer
from .modules.regime_features import RegimeFeatureEngineer
from .modules.microstructure_features import MicrostructureFeatureEngineer
from .modules.alternative_data_features import AlternativeDataFeatureEngineer

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for crypto trading - Main coordinator"""
    
    def __init__(self):
        # Initialize all feature engineering modules
        self.time_engineer = TimeFeatureEngineer()
        self.wavelet_engineer = WaveletFeatureEngineer()
        self.statistical_engineer = StatisticalFeatureEngineer()
        self.regime_engineer = RegimeFeatureEngineer()
        self.microstructure_engineer = MicrostructureFeatureEngineer()
        self.alternative_data_engineer = AlternativeDataFeatureEngineer()
        
        # PCA models for dimensionality reduction
        self.pca_models = {}
        
        logger.info("FeatureEngineer initialized with all modules")
    
    def engineer_all_features(self, df: pd.DataFrame, orderbook_data: Optional[Dict] = None) -> pd.DataFrame:
        """Engineer all features for the dataset"""
        logger.info("Starting comprehensive feature engineering")
        
        initial_features = len(df.columns)
        
        # Time-based features
        df = self.time_engineer.create_time_features(df)
        
        # Market microstructure features (HIGH IMPACT)
        if orderbook_data:
            df = self.microstructure_engineer.create_microstructure_features(df, orderbook_data)
        
        # Wavelet features
        df = self.wavelet_engineer.create_wavelet_features(df)
        
        # Statistical features
        df = self.statistical_engineer.create_statistical_features(df)
        
        # Fourier features
        df = self.regime_engineer.create_fourier_features(df)
        
        # Market regime features
        df = self.regime_engineer.create_regime_features(df)
        
        # Interaction features
        df = self.microstructure_engineer.create_interaction_features(df)
        
        # Lag features
        df = self.microstructure_engineer.create_lag_features(df)
        
        # Rolling window features
        df = self.statistical_engineer.create_rolling_features(df)
        
        # Alternative data features
        df = self.alternative_data_engineer.create_alternative_data_features(df)
        
        # Multi-timeframe features
        df = self.alternative_data_engineer.create_multi_timeframe_features(df)
        
        final_features = len(df.columns)
        logger.info(f"Feature engineering complete. Features: {initial_features} → {final_features}")
        
        return df
    
    def apply_pca(self, df: pd.DataFrame, feature_group: str, n_components: float = 0.95) -> pd.DataFrame:
        """Apply PCA to reduce dimensionality of a feature group"""
        # Select features for the group
        feature_cols = [col for col in df.columns if feature_group in col]
        
        if len(feature_cols) < 2:
            return df
        
        # Fit or transform PCA
        if feature_group not in self.pca_models:
            self.pca_models[feature_group] = PCA(n_components=n_components)
            pca_features = self.pca_models[feature_group].fit_transform(df[feature_cols].fillna(0))
        else:
            pca_features = self.pca_models[feature_group].transform(df[feature_cols].fillna(0))
        
        # Add PCA features
        n_comps = pca_features.shape[1]
        for i in range(n_comps):
            df[f'{feature_group}_pca_{i}'] = pca_features[:, i]
        
        # Optionally drop original features
        # df = df.drop(columns=feature_cols)
        
        logger.info(f"Applied PCA to {feature_group}: {len(feature_cols)} → {n_comps} components")
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for importance analysis"""
        return {
            'time': ['hour', 'day', 'month', 'quarter', 'weekend'],
            'wavelet': ['wavelet'],
            'statistical': ['return_mean', 'return_std', 'return_skew', 'return_kurt', 'entropy'],
            'regime': ['regime', 'fft'],
            'microstructure': ['spread', 'imbalance', 'flow', 'pressure', 'depth'],
            'rolling': ['min_max_ratio', 'close_position', 'volume_trend', 'efficiency_ratio'],
            'interaction': ['price_vs', 'above', 'oversold', 'overbought', 'volume_price'],
            'lag': ['lag_'],
            'alternative': ['sentiment', 'fear_greed', 'addresses', 'whale', 'vix', 'dxy'],
            'multi_timeframe': ['htf_', 'mtf_', 'stf_', 'agreement', 'divergence']
        }
    
    def select_top_features(self, df: pd.DataFrame, target_col: str, n_features: int = 50) -> List[str]:
        """Select top features based on correlation with target"""
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove target and highly correlated features
        selected_features = []
        for feature in correlations.index:
            if feature == target_col:
                continue
                
            # Check correlation with already selected features
            add_feature = True
            for selected in selected_features:
                if abs(df[feature].corr(df[selected])) > 0.95:
                    add_feature = False
                    break
            
            if add_feature:
                selected_features.append(feature)
                
            if len(selected_features) >= n_features:
                break
        
        logger.info(f"Selected top {len(selected_features)} features")
        
        return selected_features
