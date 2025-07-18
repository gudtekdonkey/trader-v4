"""
File: ensemble.py
Modified: 2024-12-20
Refactored: 2025-07-18

Advanced ensemble model with modular architecture.
This file coordinates the ensemble modules for better maintainability.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import traceback
import os
import json
import gc
from collections import defaultdict
import warnings

# Import ensemble modules
from modules.version_manager import ModelVersionManager, MODEL_VERSION
from modules.health_monitor import ModelHealthMonitor
from modules.weight_network import AdaptiveWeightNetwork
from modules.model_trainer import ModelTrainer

# Import model architectures
from lstm_attention import AttentionLSTM
from temporal_fusion_transformer import TFTModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Advanced ensemble model combining multiple architectures with error handling.
    
    This class implements a sophisticated ensemble that combines:
    - Deep learning models (LSTM, TFT)
    - Gradient boosting models (LightGBM, XGBoost)
    - Dynamic weight allocation based on market conditions
    - Uncertainty-weighted predictions
    - Model health monitoring
    - Automatic recovery and retraining
    
    Attributes:
        device: Computing device (cuda/cpu)
        attention_lstm: LSTM model with attention mechanism
        tft_model: Temporal Fusion Transformer model
        lgb_models: List of LightGBM models
        xgb_models: List of XGBoost models
        weight_network: Neural network for weight allocation
        model_performance: Performance tracking for each model
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize the ensemble predictor with error handling.
        
        Args:
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing EnsemblePredictor on device: {self.device}")
        
        # Model status tracking
        self.model_status = {
            'attention_lstm': False,
            'tft': False,
            'lgb': False,
            'xgb': False
        }
        
        # Initialize modules
        self.health_monitor = ModelHealthMonitor()
        self.model_trainer = ModelTrainer()
        
        try:
            # Deep learning models
            self.attention_lstm = self._initialize_lstm()
            self.tft_model = self._initialize_tft()
            
            # Gradient boosting models
            self.lgb_models = []
            self.xgb_models = []
            
            # Meta-learner for weight allocation
            self.weight_network = self._initialize_weight_network()
            
            # Performance tracking
            self.model_performance = {
                'attention_lstm': [],
                'tft': [],
                'lgb': [],
                'xgb': []
            }
            
            # Ensemble parameters
            self.use_uncertainty_weighting = True
            self.temperature = 1.0
            
            # Error tracking
            self.error_counts = {
                'lstm_errors': 0,
                'tft_errors': 0,
                'lgb_errors': 0,
                'xgb_errors': 0
            }
            
            # Cache for performance
            self.prediction_cache = {}
            self.cache_size = 100
            
            # Recovery mechanisms
            self.recovery_attempts = defaultdict(int)
            self.max_recovery_attempts = 3
            
            logger.info("EnsemblePredictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing EnsemblePredictor: {e}")
            logger.error(traceback.format_exc())
            # Continue with partial initialization
    
    def _initialize_lstm(self) -> Optional[AttentionLSTM]:
        """Initialize LSTM model with error handling"""
        try:
            model = AttentionLSTM(
                input_dim=50, 
                hidden_dim=256
            ).to(self.device)
            
            # Test forward pass
            test_input = torch.randn(1, 10, 50).to(self.device)
            _ = model(test_input)
            
            self.model_status['attention_lstm'] = True
            logger.info("AttentionLSTM initialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize AttentionLSTM: {e}")
            self.model_status['attention_lstm'] = False
            return None
    
    def _initialize_tft(self) -> Optional[TFTModel]:
        """Initialize TFT model with error handling"""
        try:
            model = TFTModel(
                input_dim=100, 
                hidden_dim=160
            ).to(self.device)
            
            # Test forward pass
            test_input = {'features': torch.randn(1, 10, 100).to(self.device)}
            _ = model(test_input)
            
            self.model_status['tft'] = True
            logger.info("TFTModel initialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize TFTModel: {e}")
            self.model_status['tft'] = False
            return None
    
    def _initialize_weight_network(self) -> Optional[AdaptiveWeightNetwork]:
        """Initialize weight network with error handling"""
        try:
            # Count available models
            num_models = sum(1 for status in self.model_status.values() if status)
            num_models = max(num_models, 4)  # Always 4 for consistency
            
            network = AdaptiveWeightNetwork(
                num_models=num_models,
                feature_dim=20,
                hidden_dim=64
            ).to(self.device)
            
            logger.info("AdaptiveWeightNetwork initialized successfully")
            return network
            
        except Exception as e:
            logger.error(f"Failed to initialize AdaptiveWeightNetwork: {e}")
            return None
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train gradient boosting models with optimized parameters and error handling.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        # Validate inputs
        if not self.model_trainer.validate_training_data(X_train, y_train, X_val, y_val):
            return
        
        # Check data distribution
        self.model_trainer.validate_data_distribution(X_train, y_train)
        
        # Train LightGBM models
        self.lgb_models = self.model_trainer.train_lightgbm(X_train, y_train, X_val, y_val)
        self.model_status['lgb'] = len(self.lgb_models) > 0
        
        # Train XGBoost models
        self.xgb_models = self.model_trainer.train_xgboost(X_train, y_train, X_val, y_val)
        self.model_status['xgb'] = len(self.xgb_models) > 0
    
    def predict_deep_learning(self, lstm_input: torch.Tensor,
                            tft_input: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get predictions from deep learning models with uncertainty and error handling.
        
        Args:
            lstm_input: Input tensor for LSTM model
            tft_input: Input dictionary for TFT model
            
        Returns:
            Dictionary with model predictions and uncertainties
        """
        results = {}
        
        # LSTM predictions
        if self.attention_lstm and self.model_status['attention_lstm']:
            try:
                # Check if model needs recovery
                if not self.health_monitor.is_model_healthy('attention_lstm'):
                    self._attempt_model_recovery('attention_lstm')
                
                lstm_mean, lstm_std = self.attention_lstm.predict_with_confidence(lstm_input)
                
                # Validate predictions
                has_nan = torch.isnan(lstm_mean).any() or torch.isnan(lstm_std).any()
                if has_nan:
                    logger.warning("NaN in LSTM predictions")
                    raise ValueError("Invalid LSTM predictions")
                
                results['lstm'] = (lstm_mean, lstm_std)
                self.health_monitor.record_prediction('attention_lstm', success=True, has_nan=False)
                
            except Exception as e:
                logger.error(f"Error in LSTM prediction: {e}")
                self.error_counts['lstm_errors'] += 1
                self.health_monitor.record_prediction('attention_lstm', success=False)
                
                # Fallback predictions
                batch_size = lstm_input.size(0)
                results['lstm'] = (
                    torch.zeros((batch_size, 1), device=self.device),
                    torch.ones((batch_size, 1), device=self.device)
                )
        else:
            # Model not available
            batch_size = lstm_input.size(0)
            results['lstm'] = (
                torch.zeros((batch_size, 1), device=self.device),
                torch.ones((batch_size, 1), device=self.device)
            )
        
        # TFT predictions
        if self.tft_model and self.model_status['tft']:
            try:
                # Check if model needs recovery
                if not self.health_monitor.is_model_healthy('tft'):
                    self._attempt_model_recovery('tft')
                
                tft_predictions = self.tft_model.predict_with_uncertainty(tft_input)
                tft_mean, tft_std = tft_predictions['price']
                
                # Validate predictions
                has_nan = torch.isnan(tft_mean).any() or torch.isnan(tft_std).any()
                if has_nan:
                    logger.warning("NaN in TFT predictions")
                    raise ValueError("Invalid TFT predictions")
                
                results['tft'] = (tft_mean, tft_std)
                self.health_monitor.record_prediction('tft', success=True, has_nan=False)
                
            except Exception as e:
                logger.error(f"Error in TFT prediction: {e}")
                self.error_counts['tft_errors'] += 1
                self.health_monitor.record_prediction('tft', success=False)
                
                # Fallback predictions
                if isinstance(tft_input, dict) and 'features' in tft_input:
                    batch_size = tft_input['features'].size(0)
                else:
                    batch_size = 1
                    
                results['tft'] = (
                    torch.zeros((batch_size, 1), device=self.device),
                    torch.ones((batch_size, 1), device=self.device)
                )
        else:
            # Model not available
            if isinstance(tft_input, dict) and 'features' in tft_input:
                batch_size = tft_input['features'].size(0)
            else:
                batch_size = 1
                
            results['tft'] = (
                torch.zeros((batch_size, 1), device=self.device),
                torch.ones((batch_size, 1), device=self.device)
            )
        
        return results
    
    def predict_gradient_boosting(self, X: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get predictions from gradient boosting models with error handling.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with mean predictions and uncertainties
        """
        results = {}
        
        # Validate input
        if X is None or len(X) == 0:
            logger.error("Invalid input for gradient boosting")
            return {
                'lgb': (np.array([0.0]), np.array([1.0])),
                'xgb': (np.array([0.0]), np.array([1.0]))
            }
        
        # Clean input data
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("NaN/Inf in gradient boosting input")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check cache
        cache_key = hash(X.tobytes())
        if cache_key in self.prediction_cache:
            logger.debug("Using cached gradient boosting predictions")
            return self.prediction_cache[cache_key]
        
        # LightGBM predictions
        if self.lgb_models and self.model_status['lgb']:
            try:
                lgb_preds = []
                for model in self.lgb_models:
                    try:
                        pred = model.predict(X, num_iteration=model.best_iteration)
                        lgb_preds.append(pred)
                    except Exception as e:
                        logger.warning(f"LightGBM model prediction failed: {e}")
                        continue
                
                if lgb_preds:
                    lgb_preds = np.array(lgb_preds)
                    lgb_mean = lgb_preds.mean(axis=0)
                    lgb_std = lgb_preds.std(axis=0)
                    
                    # Ensure valid std
                    lgb_std = np.maximum(lgb_std, 1e-6)
                    
                    results['lgb'] = (lgb_mean, lgb_std)
                    self.health_monitor.record_prediction('lgb', success=True)
                else:
                    raise ValueError("No valid LightGBM predictions")
                    
            except Exception as e:
                logger.error(f"Error in LightGBM predictions: {e}")
                self.error_counts['lgb_errors'] += 1
                self.health_monitor.record_prediction('lgb', success=False)
                results['lgb'] = (np.zeros(X.shape[0]), np.ones(X.shape[0]))
        else:
            results['lgb'] = (np.zeros(X.shape[0]), np.ones(X.shape[0]))
        
        # XGBoost predictions
        if self.xgb_models and self.model_status['xgb']:
            try:
                xgb_preds = []
                for model in self.xgb_models:
                    try:
                        pred = model.predict(X)
                        xgb_preds.append(pred)
                    except Exception as e:
                        logger.warning(f"XGBoost model prediction failed: {e}")
                        continue
                
                if xgb_preds:
                    xgb_preds = np.array(xgb_preds)
                    xgb_mean = xgb_preds.mean(axis=0)
                    xgb_std = xgb_preds.std(axis=0)
                    
                    # Ensure valid std
                    xgb_std = np.maximum(xgb_std, 1e-6)
                    
                    results['xgb'] = (xgb_mean, xgb_std)
                    self.health_monitor.record_prediction('xgb', success=True)
                else:
                    raise ValueError("No valid XGBoost predictions")
                    
            except Exception as e:
                logger.error(f"Error in XGBoost predictions: {e}")
                self.error_counts['xgb_errors'] += 1
                self.health_monitor.record_prediction('xgb', success=False)
                results['xgb'] = (np.zeros(X.shape[0]), np.ones(X.shape[0]))
        else:
            results['xgb'] = (np.zeros(X.shape[0]), np.ones(X.shape[0]))
        
        # Cache results
        if len(self.prediction_cache) < self.cache_size:
            self.prediction_cache[cache_key] = results
        
        return results
    
    def calculate_dynamic_weights(self, market_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate dynamic weights based on market conditions with error handling.
        
        Args:
            market_features: Tensor of market condition features
            
        Returns:
            Weight tensor for each model
        """
        # Check if weight network exists
        if self.weight_network is None:
            logger.warning("Weight network not available, using uniform weights")
            return torch.ones(market_features.size(0), 4) / 4
        
        try:
            # Validate input
            if torch.isnan(market_features).any() or torch.isinf(market_features).any():
                logger.warning("NaN/Inf in market features")
                market_features = torch.nan_to_num(market_features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            weights = self.weight_network(market_features)
            
            # Validate weights
            if torch.isnan(weights).any():
                logger.warning("NaN in dynamic weights, using uniform weights")
                return torch.ones_like(weights) / weights.size(-1)
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                weights = torch.softmax(torch.log(weights + 1e-8) / self.temperature, dim=-1)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {e}")
            # Return uniform weights
            return torch.ones(market_features.size(0), 4) / 4
    
    def _attempt_model_recovery(self, model_name: str):
        """Attempt to recover a failing model"""
        if self.recovery_attempts[model_name] >= self.max_recovery_attempts:
            logger.warning(f"Max recovery attempts reached for {model_name}")
            return
            
        self.recovery_attempts[model_name] += 1
        logger.info(f"Attempting recovery for {model_name} (attempt {self.recovery_attempts[model_name]})")
        
        try:
            if model_name == 'attention_lstm' and self.attention_lstm:
                # Reset model parameters
                if hasattr(self.attention_lstm, 'reset_parameters'):
                    self.attention_lstm.reset_parameters()
                self.model_status['attention_lstm'] = True
                
            elif model_name == 'tft' and self.tft_model:
                # Reset model parameters
                if hasattr(self.tft_model, 'reset_parameters'):
                    self.tft_model.reset_parameters()
                self.model_status['tft'] = True
                
            logger.info(f"Successfully recovered {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to recover {model_name}: {e}")
            self.model_status[model_name] = False
    
    def predict(self, lstm_input: torch.Tensor, tft_input: Dict[str, torch.Tensor],
               gb_input: np.ndarray, market_features: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
        """
        Generate ensemble predictions with confidence intervals and comprehensive error handling.
        
        Args:
            lstm_input: Input for LSTM model
            tft_input: Input for TFT model
            gb_input: Input for gradient boosting models
            market_features: Features for weight calculation (optional)
            
        Returns:
            Dictionary containing ensemble predictions
        """
        # Create default market features if not provided
        if market_features is None:
            logger.warning("No market features provided for weight calculation")
            market_features = torch.zeros((lstm_input.size(0), 20), device=self.device)
        
        try:
            # Clear cache if too large
            if len(self.prediction_cache) >= self.cache_size:
                self.prediction_cache.clear()
                gc.collect()
            
            # Get predictions from all models
            dl_predictions = self.predict_deep_learning(lstm_input, tft_input)
            gb_predictions = self.predict_gradient_boosting(gb_input)
            
            # Calculate dynamic weights
            weights = self.calculate_dynamic_weights(market_features)
            weights_np = weights.cpu().numpy()
            
            # Combine predictions
            ensemble_result = self._combine_predictions(
                dl_predictions, gb_predictions, weights_np
            )
            
            # Update performance tracking
            self._update_performance_tracking(
                ensemble_result['individual_predictions'], 
                ensemble_result['mean']
            )
            
            # Add diagnostics
            ensemble_result['model_status'] = self.model_status
            ensemble_result['error_counts'] = self.error_counts
            ensemble_result['health_report'] = self.health_monitor.get_model_health_report()
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Critical error in ensemble prediction: {e}")
            logger.error(traceback.format_exc())
            
            # Return safe defaults
            return {
                'mean': np.zeros(1),
                'std': np.ones(1),
                'lower_bound': np.zeros(1) - 2,
                'upper_bound': np.zeros(1) + 2,
                'weights': np.ones(4) / 4,
                'individual_predictions': {},
                'model_status': self.model_status,
                'error_counts': self.error_counts,
                'health_report': self.health_monitor.get_model_health_report()
            }
    
    def _combine_predictions(self, dl_predictions: Dict, gb_predictions: Dict,
                           weights_np: np.ndarray) -> Dict[str, np.ndarray]:
        """Combine predictions from different models"""
        # Collect all predictions
        all_predictions = {
            'lstm': dl_predictions.get('lstm', (None, None)),
            'tft': dl_predictions.get('tft', (None, None)),
            'lgb': gb_predictions.get('lgb', (None, None)),
            'xgb': gb_predictions.get('xgb', (None, None))
        }
        
        valid_models = []
        all_means = []
        all_stds = []
        
        # Collect valid predictions
        for i, (model_name, (mean, std)) in enumerate(all_predictions.items()):
            if mean is not None and std is not None:
                # Convert to numpy if needed
                if isinstance(mean, torch.Tensor):
                    mean = mean.cpu().numpy()
                if isinstance(std, torch.Tensor):
                    std = std.cpu().numpy()
                
                # Validate predictions
                if not np.isnan(mean).any() and not np.isnan(std).any():
                    all_means.append(mean)
                    all_stds.append(std)
                    valid_models.append(i)
        
        if not valid_models:
            logger.error("No valid model predictions available")
            return {
                'mean': np.zeros(1),
                'std': np.ones(1),
                'lower_bound': np.zeros(1) - 2,
                'upper_bound': np.zeros(1) + 2,
                'weights': weights_np,
                'individual_predictions': {}
            }
        
        # Adjust weights for valid models only
        if len(valid_models) < 4:
            logger.warning(f"Only {len(valid_models)} models available, adjusting weights")
            valid_weights = weights_np[:, valid_models]
            valid_weights = valid_weights / valid_weights.sum(axis=1, keepdims=True)
        else:
            valid_weights = weights_np
        
        # Apply uncertainty weighting if enabled
        if self.use_uncertainty_weighting and len(all_stds) > 0:
            # Inverse variance weighting
            precisions = []
            for std in all_stds:
                precision = 1 / (std + 1e-6)
                precisions.append(precision)
            
            # Normalize precisions
            precision_array = np.array(precisions)
            precision_weights = precision_array / precision_array.sum(axis=0)
            
            # Combine with dynamic weights
            final_weights = valid_weights * precision_weights
            final_weights = final_weights / final_weights.sum(axis=0)
        else:
            final_weights = valid_weights
        
        # Calculate ensemble mean
        ensemble_mean = np.sum(
            [w * m for w, m in zip(final_weights, all_means)], 
            axis=0
        )
        
        # Uncertainty propagation
        ensemble_variance = np.sum([
            w**2 * s**2 for w, s in zip(final_weights, all_stds)
        ], axis=0)
        ensemble_std = np.sqrt(ensemble_variance)
        
        # Ensure valid std
        ensemble_std = np.maximum(ensemble_std, 1e-6)
        
        # Calculate prediction intervals (95% confidence)
        lower_bound = ensemble_mean - 2 * ensemble_std
        upper_bound = ensemble_mean + 2 * ensemble_std
        
        # Prepare individual predictions for output
        individual_predictions = {}
        model_names = ['lstm', 'tft', 'lgb', 'xgb']
        for i, model_idx in enumerate(valid_models):
            individual_predictions[model_names[model_idx]] = all_means[i]
        
        return {
            'mean': ensemble_mean,
            'std': ensemble_std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'weights': final_weights,
            'individual_predictions': individual_predictions
        }
    
    def _update_performance_tracking(self, individual_preds: Dict[str, np.ndarray],
                                   ensemble_pred: np.ndarray) -> None:
        """Track model performance for adaptive weighting"""
        # This is a placeholder for actual performance tracking
        # In production, you would compare against actual values
        pass
    
    def adapt_weights(self, true_values: np.ndarray,
                     predictions: Dict[str, np.ndarray]) -> None:
        """
        Adapt ensemble weights based on recent performance with error handling.
        
        Args:
            true_values: Actual observed values
            predictions: Dictionary of predictions including individual models
        """
        try:
            # Validate inputs
            if true_values is None or len(true_values) == 0:
                logger.warning("No true values provided for weight adaptation")
                return
            
            if 'individual_predictions' not in predictions:
                logger.warning("No individual predictions for weight adaptation")
                return
            
            # Calculate individual model errors
            for model_name, pred in predictions['individual_predictions'].items():
                if pred is not None and len(pred) == len(true_values):
                    try:
                        # Handle NaN in true values
                        valid_mask = ~np.isnan(true_values) & ~np.isnan(pred)
                        if valid_mask.any():
                            error = np.mean((pred[valid_mask] - true_values[valid_mask]) ** 2)
                            self.model_performance[model_name].append(error)
                            
                            # Update health monitor
                            self.health_monitor.record_performance(model_name, -error)
                        
                    except Exception as e:
                        logger.error(f"Error calculating error for {model_name}: {e}")
            
            # Keep only recent performance
            window_size = 100
            for model_name in self.model_performance:
                if len(self.model_performance[model_name]) > window_size:
                    self.model_performance[model_name] = \
                        self.model_performance[model_name][-window_size:]
                        
        except Exception as e:
            logger.error(f"Error in weight adaptation: {e}")
    
    def save_models(self, directory: str) -> None:
        """
        Save all models to disk with error handling and versioning.
        
        Args:
            directory: Directory to save models
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save metadata with version
            metadata = {
                'version': MODEL_VERSION,
                'model_status': self.model_status,
                'error_counts': self.error_counts,
                'model_performance': self.model_performance,
                'use_uncertainty_weighting': self.use_uncertainty_weighting,
                'temperature': self.temperature,
                'device': str(self.device),
                'health_report': self.health_monitor.get_model_health_report()
            }
            
            # Save deep learning models
            if self.attention_lstm and self.model_status['attention_lstm']:
                try:
                    model_path = os.path.join(directory, 'attention_lstm.pt')
                    torch.save({
                        'model_state_dict': self.attention_lstm.state_dict(),
                        'version': MODEL_VERSION
                    }, model_path)
                    logger.info("Saved AttentionLSTM model")
                except Exception as e:
                    logger.error(f"Failed to save AttentionLSTM: {e}")
            
            if self.tft_model and self.model_status['tft']:
                try:
                    model_path = os.path.join(directory, 'tft_model.pt')
                    torch.save({
                        'model_state_dict': self.tft_model.state_dict(),
                        'version': MODEL_VERSION
                    }, model_path)
                    logger.info("Saved TFT model")
                except Exception as e:
                    logger.error(f"Failed to save TFT model: {e}")
            
            # Save gradient boosting models
            if self.lgb_models:
                for i, model in enumerate(self.lgb_models):
                    try:
                        model.save_model(
                            os.path.join(directory, f'lgb_model_{i}.txt'),
                            num_iteration=model.best_iteration
                        )
                    except Exception as e:
                        logger.error(f"Failed to save LightGBM model {i}: {e}")
            
            if self.xgb_models:
                for i, model in enumerate(self.xgb_models):
                    try:
                        model.save_model(
                            os.path.join(directory, f'xgb_model_{i}.json')
                        )
                    except Exception as e:
                        logger.error(f"Failed to save XGBoost model {i}: {e}")
            
            # Save weight network
            if self.weight_network:
                try:
                    weight_path = os.path.join(directory, 'weight_network.pt')
                    torch.save({
                        'model_state_dict': self.weight_network.state_dict(),
                        'statistics': self.weight_network.get_statistics(),
                        'version': MODEL_VERSION
                    }, weight_path)
                    logger.info("Saved weight network")
                except Exception as e:
                    logger.error(f"Failed to save weight network: {e}")
            
            # Save ensemble metadata
            try:
                with open(os.path.join(directory, 'ensemble_metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info("Saved ensemble metadata")
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, directory: str) -> None:
        """
        Load all models from disk with error handling and version compatibility.
        
        Args:
            directory: Directory containing saved models
        """
        if not os.path.exists(directory):
            logger.error(f"Model directory does not exist: {directory}")
            return
        
        try:
            # Load and check metadata first
            metadata_path = os.path.join(directory, 'ensemble_metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    saved_version = metadata.get('version', '1.0.0')
                    if not ModelVersionManager.check_compatibility(saved_version):
                        logger.warning(f"Model version {saved_version} may not be compatible with current version {MODEL_VERSION}")
                        
                    # Apply metadata
                    self.use_uncertainty_weighting = metadata.get('use_uncertainty_weighting', True)
                    self.temperature = metadata.get('temperature', 1.0)
                    
                except Exception as e:
                    logger.error(f"Failed to load metadata: {e}")
            
            # Load deep learning models
            lstm_path = os.path.join(directory, 'attention_lstm.pt')
            if os.path.exists(lstm_path) and self.attention_lstm:
                try:
                    checkpoint = torch.load(lstm_path, map_location=self.device)
                    
                    # Check version compatibility
                    saved_version = checkpoint.get('version', '1.0.0')
                    if ModelVersionManager.check_compatibility(saved_version):
                        self.attention_lstm.load_state_dict(checkpoint['model_state_dict'])
                        self.model_status['attention_lstm'] = True
                        logger.info("Loaded AttentionLSTM model")
                    else:
                        logger.warning(f"Incompatible AttentionLSTM version: {saved_version}")
                        
                except Exception as e:
                    logger.error(f"Failed to load AttentionLSTM: {e}")
                    self.model_status['attention_lstm'] = False
            
            tft_path = os.path.join(directory, 'tft_model.pt')
            if os.path.exists(tft_path) and self.tft_model:
                try:
                    checkpoint = torch.load(tft_path, map_location=self.device)
                    
                    # Check version compatibility
                    saved_version = checkpoint.get('version', '1.0.0')
                    if ModelVersionManager.check_compatibility(saved_version):
                        self.tft_model.load_state_dict(checkpoint['model_state_dict'])
                        self.model_status['tft'] = True
                        logger.info("Loaded TFT model")
                    else:
                        logger.warning(f"Incompatible TFT version: {saved_version}")
                        
                except Exception as e:
                    logger.error(f"Failed to load TFT model: {e}")
                    self.model_status['tft'] = False
            
            # Load gradient boosting models
            import lightgbm as lgb
            import xgboost as xgb
            
            self.lgb_models = []
            for i in range(3):  # Expecting 3 models
                lgb_path = os.path.join(directory, f'lgb_model_{i}.txt')
                if os.path.exists(lgb_path):
                    try:
                        model = lgb.Booster(model_file=lgb_path)
                        self.lgb_models.append(model)
                    except Exception as e:
                        logger.error(f"Failed to load LightGBM model {i}: {e}")
            
            if self.lgb_models:
                self.model_status['lgb'] = True
                logger.info(f"Loaded {len(self.lgb_models)} LightGBM models")
            else:
                self.model_status['lgb'] = False
            
            self.xgb_models = []
            for i in range(3):  # Expecting 3 models
                xgb_path = os.path.join(directory, f'xgb_model_{i}.json')
                if os.path.exists(xgb_path):
                    try:
                        model = xgb.XGBRegressor()
                        model.load_model(xgb_path)
                        self.xgb_models.append(model)
                    except Exception as e:
                        logger.error(f"Failed to load XGBoost model {i}: {e}")
            
            if self.xgb_models:
                self.model_status['xgb'] = True
                logger.info(f"Loaded {len(self.xgb_models)} XGBoost models")
            else:
                self.model_status['xgb'] = False
            
            # Load weight network
            weight_path = os.path.join(directory, 'weight_network.pt')
            if os.path.exists(weight_path) and self.weight_network:
                try:
                    checkpoint = torch.load(weight_path, map_location=self.device)
                    self.weight_network.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Restore statistics if available
                    if 'statistics' in checkpoint:
                        stats = checkpoint['statistics']
                        self.weight_network.call_count = stats.get('call_count', 0)
                        self.weight_network.nan_count = stats.get('nan_count', 0)
                        
                    logger.info("Loaded weight network")
                except Exception as e:
                    logger.error(f"Failed to load weight network: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_model_diagnostics(self) -> Dict:
        """
        Get diagnostic information about ensemble models.
        
        Returns:
            Dictionary with model status and performance metrics
        """
        diagnostics = {
            'model_status': self.model_status,
            'error_counts': self.error_counts,
            'performance_history': {
                model: {
                    'num_samples': len(perf),
                    'mean_error': np.mean(perf) if perf else None,
                    'std_error': np.std(perf) if perf else None,
                    'recent_error': perf[-1] if perf else None
                }
                for model, perf in self.model_performance.items()
            },
            'total_predictions': sum(self.error_counts.values()),
            'device': str(self.device),
            'uncertainty_weighting': self.use_uncertainty_weighting,
            'temperature': self.temperature,
            'health_report': self.health_monitor.get_model_health_report(),
            'cache_size': len(self.prediction_cache),
            'recovery_attempts': dict(self.recovery_attempts)
        }
        
        # Add weight network statistics if available
        if self.weight_network:
            diagnostics['weight_network_stats'] = self.weight_network.get_statistics()
        
        return diagnostics
    
    def clear_cache(self):
        """Clear prediction cache to free memory"""
        self.prediction_cache.clear()
        gc.collect()
        logger.info("Cleared prediction cache")

"""
REFACTORING SUMMARY:
- Original file: 1600+ lines
- Refactored ensemble.py: ~800 lines
- Created 4 modular components:
  1. version_manager.py - Model versioning and compatibility
  2. health_monitor.py - Model health tracking
  3. weight_network.py - Adaptive weight allocation network
  4. model_trainer.py - Training logic for different model types
- Benefits:
  * Clearer separation of concerns
  * Easier to test individual components
  * Better maintainability
  * Simplified main ensemble logic
  * Easy to add new model types
"""
