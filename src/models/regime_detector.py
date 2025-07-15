"""
Market Regime Detection System for Cryptocurrency Trading

This module implements a sophisticated market regime detection system using:
- Hidden Markov Models (HMM) for regime identification
- Neural networks for regime classification
- Technical indicators and market microstructure features
- Dynamic trading parameter adjustment based on regime
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from typing import Tuple, List, Dict, Optional
import pandas as pd


class MarketRegimeDetector:
    """
    Detect and classify market regimes using HMM and neural networks.
    
    This class identifies different market regimes (e.g., low volatility,
    trending, high volatility, extreme conditions) and provides trading
    recommendations based on the current regime.
    
    Features:
    - Automatic regime detection using Hidden Markov Models
    - Neural network classification for regime prediction
    - Regime-specific trading parameters
    - Transition probability analysis
    
    Attributes:
        n_regimes: Number of distinct market regimes
        hmm_model: Hidden Markov Model for regime detection
        regime_classifier: Neural network for regime classification
        regime_stats: Statistical characteristics of each regime
        current_regime: Current detected market regime
        regime_history: Historical regime transitions
    """
    
    def __init__(self, n_regimes: int = 4):
        """
        Initialize the market regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect (default: 4)
        """
        self.n_regimes = n_regimes
        
        # HMM for regime detection
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        # Neural network for regime classification
        self.regime_classifier = self._build_classifier()
        
        # Regime characteristics
        self.regime_stats = {}
        self.current_regime = None
        self.regime_history = []
        
        # Feature scaler
        self.scaler = StandardScaler()
    
    def _build_classifier(self) -> nn.Module:
        """
        Build neural network for regime classification.
        
        Returns:
            Neural network module for classification
        """
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.n_regimes),
            nn.Softmax(dim=-1)
        )
    
    def extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for regime detection.
        
        Args:
            data: DataFrame with OHLCV and additional market data
            
        Returns:
            Feature matrix for regime detection
        """
        features = []
        
        # Price-based features
        returns = data['close'].pct_change()
        features.append(returns.rolling(20).mean())   # Trend
        features.append(returns.rolling(20).std())    # Volatility
        features.append(returns.rolling(20).skew())   # Skewness
        features.append(returns.rolling(20).kurt())   # Kurtosis
        
        # Volume features
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
        features.append(volume_ratio)
        
        # Technical indicators for regime
        features.append(self._calculate_rsi(data['close'], 14))
        features.append(self._calculate_adx(data, 14))
        
        # Market microstructure (if available)
        if 'bid_ask_spread' in data.columns:
            features.append(data['bid_ask_spread'].rolling(20).mean())
        
        # Order flow imbalance (if available)
        if 'order_flow_imbalance' in data.columns:
            features.append(data['order_flow_imbalance'].rolling(20).mean())
        
        # Convert to numpy array
        feature_matrix = np.column_stack([
            f.fillna(0).values for f in features
        ])
        
        return feature_matrix
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index.
        
        Args:
            data: DataFrame with high, low, close prices
            period: ADX period
            
        Returns:
            ADX values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional movements
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = up_move.where(
            (up_move > down_move) & (up_move > 0), 0
        )
        neg_dm = down_move.where(
            (down_move > up_move) & (down_move > 0), 0
        )
        
        # Directional indicators
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        # ADX calculation
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def fit(self, historical_data: pd.DataFrame) -> 'MarketRegimeDetector':
        """
        Fit regime detection models on historical data.
        
        Args:
            historical_data: Historical market data
            
        Returns:
            Self for method chaining
        """
        # Extract features
        features = self.extract_regime_features(historical_data)
        
        # Remove NaN values
        valid_idx = ~np.isnan(features).any(axis=1)
        features_clean = features[valid_idx]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # Fit HMM model
        self.hmm_model.fit(features_scaled)
        
        # Get regime labels
        regime_labels = self.hmm_model.predict(features_scaled)
        
        # Calculate regime statistics
        returns = historical_data['close'].pct_change()[valid_idx]
        
        for regime in range(self.n_regimes):
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            self.regime_stats[regime] = {
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'frequency': regime_mask.sum() / len(regime_labels),
                'avg_duration': self._calculate_avg_duration(regime_labels, regime)
            }
        
        # Sort regimes by volatility (0 = lowest vol, n-1 = highest vol)
        sorted_regimes = sorted(
            self.regime_stats.items(), 
            key=lambda x: x[1]['volatility']
        )
        
        # Create mapping from old to new regime numbers
        self.regime_mapping = {
            old: new for new, (old, _) in enumerate(sorted_regimes)
        }
        
        # Update regime stats with new mapping
        new_stats = {}
        for old_regime, new_regime in self.regime_mapping.items():
            new_stats[new_regime] = self.regime_stats[old_regime]
        self.regime_stats = new_stats
        
        return self
    
    def _calculate_avg_duration(
        self, 
        regime_sequence: np.ndarray, 
        regime: int
    ) -> float:
        """
        Calculate average duration of a regime.
        
        Args:
            regime_sequence: Sequence of regime labels
            regime: Regime to calculate duration for
            
        Returns:
            Average duration in periods
        """
        durations = []
        current_duration = 0
        
        for r in regime_sequence:
            if r == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def detect_regime(self, current_data: pd.DataFrame) -> Dict[str, any]:
        """
        Detect current market regime.
        
        Args:
            current_data: Current market data
            
        Returns:
            Dictionary with regime information and trading recommendations
        """
        # Extract features
        features = self.extract_regime_features(current_data)
        
        # Get last valid row
        last_features = features[-1:]
        
        # Scale features
        features_scaled = self.scaler.transform(last_features)
        
        # Predict regime
        regime_probs = self.hmm_model.predict_proba(features_scaled)[0]
        current_regime = np.argmax(regime_probs)
        
        # Map to sorted regime
        current_regime = self.regime_mapping.get(current_regime, current_regime)
        
        # Update history
        self.current_regime = current_regime
        self.regime_history.append({
            'timestamp': current_data.index[-1],
            'regime': current_regime,
            'probabilities': regime_probs
        })
        
        # Get regime characteristics
        regime_info = self.regime_stats[current_regime].copy()
        regime_info['regime'] = current_regime
        regime_info['confidence'] = regime_probs[current_regime]
        regime_info['regime_probs'] = regime_probs
        
        # Add regime names
        regime_names = [
            'Low Volatility',
            'Normal',
            'High Volatility',
            'Extreme Volatility'
        ]
        regime_info['name'] = (
            regime_names[current_regime] 
            if current_regime < len(regime_names) 
            else f'Regime {current_regime}'
        )
        
        # Trading recommendations based on regime
        regime_info['trading_mode'] = self._get_trading_mode(current_regime)
        
        return regime_info
    
    def _get_trading_mode(self, regime: int) -> Dict[str, any]:
        """
        Get trading parameters based on regime.
        
        Args:
            regime: Current regime number
            
        Returns:
            Dictionary with trading parameters
        """
        trading_modes = {
            0: {  # Low volatility
                'name': 'Range Trading',
                'position_size_multiplier': 1.5,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.015,
                'preferred_strategies': ['mean_reversion', 'market_making'],
                'max_leverage': 5
            },
            1: {  # Normal
                'name': 'Trend Following',
                'position_size_multiplier': 1.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.03,
                'preferred_strategies': ['momentum', 'trend_following'],
                'max_leverage': 3
            },
            2: {  # High volatility
                'name': 'Reduced Risk',
                'position_size_multiplier': 0.5,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.05,
                'preferred_strategies': ['momentum', 'breakout'],
                'max_leverage': 2
            },
            3: {  # Extreme volatility
                'name': 'Risk Off',
                'position_size_multiplier': 0.25,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.08,
                'preferred_strategies': ['arbitrage', 'market_neutral'],
                'max_leverage': 1
            }
        }
        
        return trading_modes.get(regime, trading_modes[1])
    
    def get_regime_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition probability matrix.
        
        Returns:
            Transition matrix where element (i,j) is P(next=j|current=i)
        """
        return self.hmm_model.transmat_
    
    def predict_next_regime(
        self, 
        current_regime: int, 
        steps: int = 1
    ) -> Tuple[int, float]:
        """
        Predict next regime and probability.
        
        Args:
            current_regime: Current regime number
            steps: Number of steps ahead to predict
            
        Returns:
            Tuple of (most likely regime, confidence)
        """
        transition_matrix = self.get_regime_transition_matrix()
        
        # Current regime probability vector
        current_probs = np.zeros(self.n_regimes)
        current_probs[current_regime] = 1.0
        
        # Propagate probabilities
        for _ in range(steps):
            current_probs = current_probs @ transition_matrix
        
        # Most likely next regime
        next_regime = np.argmax(current_probs)
        confidence = current_probs[next_regime]
        
        return next_regime, confidence