"""
File: regime_detector.py
Modified: 2025-07-18
Changes Summary:
- Modularized into separate components for better maintainability
- Main regime detection coordination remains here
- Components moved to modules/ directory
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from typing import Tuple, List, Dict, Optional
import pandas as pd
import logging
import warnings
import traceback

# Import modularized components
from .modules.feature_extractor import FeatureExtractor
from .modules.regime_classifier import RegimeClassifier
from .modules.trading_mode_manager import TradingModeManager
from .modules.regime_analyzer import RegimeAnalyzer
from .modules.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress hmmlearn warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class MarketRegimeDetector:
    """
    Detect and classify market regimes using HMM and neural networks with error handling.
    
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
        Initialize the market regime detector with validation.
        
        Args:
            n_regimes: Number of market regimes to detect (default: 4)
        """
        # Validate inputs
        if not isinstance(n_regimes, int) or n_regimes < 2:
            logger.warning(f"Invalid n_regimes {n_regimes}, using default 4")
            n_regimes = 4
        if n_regimes > 10:
            logger.warning(f"Large n_regimes {n_regimes} may lead to overfitting")
        
        self.n_regimes = n_regimes
        self.is_fitted = False
        
        try:
            # Initialize components
            self.feature_extractor = FeatureExtractor()
            self.technical_indicators = TechnicalIndicators()
            self.regime_analyzer = RegimeAnalyzer(n_regimes)
            self.trading_mode_manager = TradingModeManager()
            
            # HMM for regime detection
            self.hmm_model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                init_params='',  # We'll initialize manually
                params='stmc'    # Update all parameters
            )
            
            # Initialize HMM parameters to prevent convergence issues
            self._initialize_hmm_params()
            
            # Neural network for regime classification
            self.regime_classifier = RegimeClassifier(n_regimes)
            
            # Regime characteristics
            self.regime_stats = {}
            self.current_regime = None
            self.regime_history = []
            
            # Feature scaler
            self.scaler = StandardScaler()
            
            # Error tracking
            self.error_stats = {
                'feature_extraction_errors': 0,
                'detection_errors': 0,
                'fitting_errors': 0
            }
            
            logger.info(f"MarketRegimeDetector initialized with {n_regimes} regimes")
            
        except Exception as e:
            logger.error(f"Failed to initialize MarketRegimeDetector: {e}")
            raise
    
    def _initialize_hmm_params(self):
        """Initialize HMM parameters for better convergence"""
        try:
            # Initialize transition matrix (slight preference for staying in same state)
            self.hmm_model.transmat_ = np.full(
                (self.n_regimes, self.n_regimes), 
                0.1 / (self.n_regimes - 1)
            )
            np.fill_diagonal(self.hmm_model.transmat_, 0.9)
            
            # Initialize start probabilities (uniform)
            self.hmm_model.startprob_ = np.ones(self.n_regimes) / self.n_regimes
            
        except Exception as e:
            logger.error(f"Error initializing HMM parameters: {e}")
    
    def extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for regime detection with comprehensive error handling.
        
        Args:
            data: DataFrame with OHLCV and additional market data
            
        Returns:
            Feature matrix for regime detection
        """
        try:
            return self.feature_extractor.extract_features(data, self.technical_indicators)
        except Exception as e:
            logger.error(f"Critical error in feature extraction: {e}")
            logger.error(traceback.format_exc())
            self.error_stats['feature_extraction_errors'] += 1
            return np.array([])
    
    def fit(self, historical_data: pd.DataFrame) -> 'MarketRegimeDetector':
        """
        Fit regime detection models on historical data with error handling.
        
        Args:
            historical_data: Historical market data
            
        Returns:
            Self for method chaining
        """
        try:
            # Validate input
            if historical_data is None or historical_data.empty:
                logger.error("Cannot fit on empty data")
                return self
            
            if len(historical_data) < 100:
                logger.warning(f"Limited data for regime detection: {len(historical_data)} rows")
            
            # Extract features
            features = self.extract_regime_features(historical_data)
            
            if features.size == 0:
                logger.error("No features extracted for fitting")
                return self
            
            # Remove NaN values
            valid_idx = ~np.isnan(features).any(axis=1)
            if not valid_idx.any():
                logger.error("All features contain NaN values")
                return self
            
            features_clean = features[valid_idx]
            
            # Check if we have enough data
            if len(features_clean) < self.n_regimes * 10:
                logger.warning(f"Very limited data for {self.n_regimes} regimes")
            
            # Scale features
            try:
                features_scaled = self.scaler.fit_transform(features_clean)
            except Exception as e:
                logger.error(f"Error scaling features: {e}")
                features_scaled = features_clean
            
            # Fit HMM model with multiple attempts
            fitted = False
            for attempt in range(3):
                try:
                    # Set random state for reproducibility
                    np.random.seed(42 + attempt)
                    
                    self.hmm_model.fit(features_scaled)
                    fitted = True
                    logger.info(f"HMM model fitted successfully on attempt {attempt + 1}")
                    break
                    
                except Exception as e:
                    logger.warning(f"HMM fitting attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        # Try with fewer regimes
                        self.n_regimes = max(2, self.n_regimes - 1)
                        self.hmm_model = hmm.GaussianHMM(
                            n_components=self.n_regimes,
                            covariance_type="diag",  # Simpler covariance
                            n_iter=50,
                            random_state=42 + attempt
                        )
            
            if not fitted:
                logger.error("Failed to fit HMM model after 3 attempts")
                self.error_stats['fitting_errors'] += 1
                return self
            
            # Get regime labels
            try:
                regime_labels = self.hmm_model.predict(features_scaled)
            except Exception as e:
                logger.error(f"Error predicting regimes: {e}")
                return self
            
            # Calculate regime statistics
            self.regime_stats = self.regime_analyzer.calculate_regime_statistics(
                historical_data[valid_idx], 
                regime_labels
            )
            
            # Create regime mapping
            self.regime_mapping = self.regime_analyzer.create_regime_mapping(self.regime_stats)
            
            # Update regime stats with new mapping
            new_stats = {}
            for old_regime, new_regime in self.regime_mapping.items():
                new_stats[new_regime] = self.regime_stats[old_regime]
            self.regime_stats = new_stats
            
            self.is_fitted = True
            logger.info("Regime detection model fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Critical error in fit method: {e}")
            logger.error(traceback.format_exc())
            self.error_stats['fitting_errors'] += 1
            return self
    
    def detect_regime(self, current_data: pd.DataFrame) -> Dict[str, any]:
        """
        Detect current market regime with comprehensive error handling.
        
        Args:
            current_data: Current market data
            
        Returns:
            Dictionary with regime information and trading recommendations
        """
        # Check if model is fitted
        if not self.is_fitted:
            logger.warning("Model not fitted, returning default regime")
            return self._get_default_regime()
        
        try:
            # Validate input
            if current_data is None or current_data.empty:
                logger.warning("Empty data for regime detection")
                return self._get_default_regime()
            
            # Extract features
            features = self.extract_regime_features(current_data)
            
            if features.size == 0:
                logger.warning("No features extracted")
                return self._get_default_regime()
            
            # Get last valid row
            last_features = features[-1:]
            
            # Check for valid features
            if np.isnan(last_features).any():
                logger.warning("NaN in current features")
                last_features = np.nan_to_num(last_features, nan=0.0)
            
            # Scale features
            try:
                features_scaled = self.scaler.transform(last_features)
            except Exception as e:
                logger.error(f"Error scaling features: {e}")
                features_scaled = last_features
            
            # Predict regime
            try:
                regime_probs = self.hmm_model.predict_proba(features_scaled)[0]
                current_regime = np.argmax(regime_probs)
            except Exception as e:
                logger.error(f"Error predicting regime: {e}")
                self.error_stats['detection_errors'] += 1
                return self._get_default_regime()
            
            # Map to sorted regime
            current_regime = self.regime_mapping.get(current_regime, current_regime)
            
            # Update history
            self.current_regime = current_regime
            self.regime_history.append({
                'timestamp': current_data.index[-1] if not current_data.empty else pd.Timestamp.now(),
                'regime': current_regime,
                'probabilities': regime_probs
            })
            
            # Limit history size
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            # Get regime characteristics
            regime_info = self.regime_stats.get(current_regime, {}).copy()
            regime_info['regime'] = current_regime
            regime_info['confidence'] = float(regime_probs[current_regime])
            regime_info['regime_probs'] = regime_probs.tolist()
            
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
            regime_info['trading_mode'] = self.trading_mode_manager.get_trading_mode(current_regime)
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Critical error in regime detection: {e}")
            logger.error(traceback.format_exc())
            self.error_stats['detection_errors'] += 1
            return self._get_default_regime()
    
    def _get_default_regime(self) -> Dict[str, any]:
        """Get default regime for error cases"""
        return {
            'regime': 1,
            'name': 'Normal',
            'confidence': 0.5,
            'regime_probs': [0.25] * self.n_regimes,
            'mean_return': 0.0,
            'volatility': 0.02,
            'frequency': 0.25,
            'avg_duration': 24.0,
            'trading_mode': self.trading_mode_manager.get_trading_mode(1)
        }
    
    def get_regime_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition probability matrix with error handling.
        
        Returns:
            Transition matrix where element (i,j) is P(next=j|current=i)
        """
        try:
            if hasattr(self.hmm_model, 'transmat_') and self.hmm_model.transmat_ is not None:
                return self.hmm_model.transmat_.copy()
            else:
                logger.warning("Transition matrix not available")
                # Return uniform transition matrix
                return np.ones((self.n_regimes, self.n_regimes)) / self.n_regimes
                
        except Exception as e:
            logger.error(f"Error getting transition matrix: {e}")
            return np.ones((self.n_regimes, self.n_regimes)) / self.n_regimes
    
    def predict_next_regime(
        self, 
        current_regime: int, 
        steps: int = 1
    ) -> Tuple[int, float]:
        """
        Predict next regime and probability with error handling.
        
        Args:
            current_regime: Current regime number
            steps: Number of steps ahead to predict
            
        Returns:
            Tuple of (most likely regime, confidence)
        """
        return self.regime_analyzer.predict_next_regime(
            current_regime, 
            steps, 
            self.get_regime_transition_matrix()
        )
    
    def get_regime_diagnostics(self) -> Dict:
        """
        Get diagnostic information about regime detection.
        
        Returns:
            Dictionary with regime detection diagnostics
        """
        diagnostics = {
            'is_fitted': self.is_fitted,
            'n_regimes': self.n_regimes,
            'current_regime': self.current_regime,
            'regime_stats': self.regime_stats,
            'error_stats': self.error_stats,
            'history_length': len(self.regime_history),
            'transition_matrix': self.get_regime_transition_matrix().tolist() if self.is_fitted else None
        }
        
        # Add recent regime changes
        if len(self.regime_history) > 1:
            recent_changes = []
            for i in range(1, min(10, len(self.regime_history))):
                if self.regime_history[-i]['regime'] != self.regime_history[-i-1]['regime']:
                    recent_changes.append({
                        'from': self.regime_history[-i-1]['regime'],
                        'to': self.regime_history[-i]['regime'],
                        'timestamp': self.regime_history[-i]['timestamp']
                    })
            diagnostics['recent_regime_changes'] = recent_changes
        
        return diagnostics

"""
MODULARIZATION_SUMMARY:
- Original file: 900+ lines
- Main file: ~400 lines (core coordination)
- Modules created:
  - feature_extractor.py: Feature extraction logic
  - regime_classifier.py: Neural network classifier
  - trading_mode_manager.py: Trading mode recommendations
  - regime_analyzer.py: Regime analysis and statistics
  - technical_indicators.py: Technical indicator calculations
- Benefits:
  - Clearer separation of concerns
  - Easier testing of individual components
  - Better code reusability
  - Simplified maintenance
"""
