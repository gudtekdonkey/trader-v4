import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from sklearn.base import BaseEstimator
import lightgbm as lgb
import xgboost as xgb
from .lstm_attention import AttentionLSTM
from .temporal_fusion_transformer import TFTModel

class AdaptiveWeightNetwork(nn.Module):
    """Neural network for dynamic ensemble weight allocation"""
    
    def __init__(self, num_models: int, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_models)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        weights = self.softmax(self.fc3(x))
        return weights

class EnsemblePredictor:
    """Advanced ensemble model combining multiple architectures"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Deep learning models
        self.attention_lstm = AttentionLSTM(input_dim=50, hidden_dim=256).to(self.device)
        self.tft_model = TFTModel(input_dim=100, hidden_dim=160).to(self.device)
        
        # Gradient boosting models
        self.lgb_models = []
        self.xgb_models = []
        
        # Meta-learner for weight allocation
        self.weight_network = AdaptiveWeightNetwork(
            num_models=4,  # LSTM, TFT, LGB, XGB
            feature_dim=20,
            hidden_dim=64
        ).to(self.device)
        
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
        
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray, 
                               X_val: np.ndarray, y_val: np.ndarray):
        """Train gradient boosting models"""
        
        # LightGBM parameters optimized for crypto
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_child_samples': 20
        }
        
        # Train multiple LightGBM models with different seeds
        for seed in [42, 123, 456]:
            lgb_params['seed'] = seed
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            self.lgb_models.append(model)
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': 4
        }
        
        # Train XGBoost
        for seed in [42, 123, 456]:
            xgb_params['random_state'] = seed
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            self.xgb_models.append(model)
    
    def predict_deep_learning(self, lstm_input: torch.Tensor, 
                            tft_input: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Get predictions from deep learning models with uncertainty"""
        
        # LSTM predictions
        lstm_mean, lstm_std = self.attention_lstm.predict_with_confidence(lstm_input)
        
        # TFT predictions
        tft_predictions = self.tft_model.predict_with_uncertainty(tft_input)
        tft_mean, tft_std = tft_predictions['price']
        
        return {
            'lstm': (lstm_mean, lstm_std),
            'tft': (tft_mean, tft_std)
        }
    
    def predict_gradient_boosting(self, X: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get predictions from gradient boosting models"""
        
        # LightGBM predictions
        lgb_preds = np.array([model.predict(X) for model in self.lgb_models])
        lgb_mean = lgb_preds.mean(axis=0)
        lgb_std = lgb_preds.std(axis=0)
        
        # XGBoost predictions
        xgb_preds = np.array([model.predict(X) for model in self.xgb_models])
        xgb_mean = xgb_preds.mean(axis=0)
        xgb_std = xgb_preds.std(axis=0)
        
        return {
            'lgb': (lgb_mean, lgb_std),
            'xgb': (xgb_mean, xgb_std)
        }
    
    def calculate_dynamic_weights(self, market_features: torch.Tensor) -> torch.Tensor:
        """Calculate dynamic weights based on market conditions"""
        return self.weight_network(market_features)
    
    def ensemble_predict(self, lstm_input: torch.Tensor, tft_input: Dict[str, torch.Tensor],
                        gb_input: np.ndarray, market_features: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate ensemble predictions with confidence intervals"""
        
        # Get predictions from all models
        dl_predictions = self.predict_deep_learning(lstm_input, tft_input)
        gb_predictions = self.predict_gradient_boosting(gb_input)
        
        # Calculate dynamic weights
        weights = self.calculate_dynamic_weights(market_features)
        weights_np = weights.cpu().numpy()
        
        # Combine predictions
        all_means = [
            dl_predictions['lstm'][0].cpu().numpy(),
            dl_predictions['tft'][0].cpu().numpy(),
            gb_predictions['lgb'][0],
            gb_predictions['xgb'][0]
        ]
        
        all_stds = [
            dl_predictions['lstm'][1].cpu().numpy(),
            dl_predictions['tft'][1].cpu().numpy(),
            gb_predictions['lgb'][1],
            gb_predictions['xgb'][1]
        ]
        
        # Weighted ensemble with uncertainty
        if self.use_uncertainty_weighting:
            # Inverse variance weighting
            precisions = [1 / (std + 1e-6) for std in all_stds]
            precision_weights = np.array(precisions) / np.sum(precisions)
            
            # Combine with dynamic weights
            final_weights = weights_np * precision_weights
            final_weights = final_weights / final_weights.sum()
        else:
            final_weights = weights_np
        
        # Calculate ensemble mean and uncertainty
        ensemble_mean = np.sum([w * m for w, m in zip(final_weights, all_means)], axis=0)
        
        # Uncertainty propagation
        ensemble_variance = np.sum([
            w**2 * s**2 for w, s in zip(final_weights, all_stds)
        ], axis=0)
        ensemble_std = np.sqrt(ensemble_variance)
        
        # Calculate prediction intervals
        lower_bound = ensemble_mean - 2 * ensemble_std
        upper_bound = ensemble_mean + 2 * ensemble_std
        
        # Update performance tracking
        self._update_performance_tracking(all_means, ensemble_mean)
        
        return {
            'mean': ensemble_mean,
            'std': ensemble_std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'weights': final_weights,
            'individual_predictions': {
                'lstm': all_means[0],
                'tft': all_means[1],
                'lgb': all_means[2],
                'xgb': all_means[3]
            }
        }
    
    def _update_performance_tracking(self, individual_preds: List[np.ndarray], 
                                   ensemble_pred: np.ndarray):
        """Track model performance for adaptive weighting"""
        # Implementation would track prediction accuracy over time
        pass
    
    def adapt_weights(self, true_values: np.ndarray, predictions: Dict[str, np.ndarray]):
        """Adapt ensemble weights based on recent performance"""
        # Calculate individual model errors
        for model_name, pred in predictions['individual_predictions'].items():
            error = np.mean((pred - true_values) ** 2)
            self.model_performance[model_name].append(error)
        
        # Keep only recent performance
        window_size = 100
        for model_name in self.model_performance:
            if len(self.model_performance[model_name]) > window_size:
                self.model_performance[model_name] = self.model_performance[model_name][-window_size:]
