"""
Model trainer module for ensemble predictor.
Handles training of different model types.
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from typing import List, Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Handles training for different model types in the ensemble"""
    
    def __init__(self):
        self.lgb_params = {
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
            'min_child_samples': 20,
            'force_col_wise': True
        }
        
        self.xgb_params = {
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
            'n_jobs': 4,
            'tree_method': 'hist'
        }
        
    def validate_training_data(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: Optional[np.ndarray] = None, 
                             y_val: Optional[np.ndarray] = None) -> bool:
        """Validate training data"""
        # Check for None
        if X_train is None or y_train is None:
            logger.error("Training data is None")
            return False
        
        # Check for empty data
        if len(X_train) == 0 or len(y_train) == 0:
            logger.error("Empty training data")
            return False
        
        # Check shape consistency
        if X_train.shape[0] != y_train.shape[0]:
            logger.error(f"Shape mismatch: X_train {X_train.shape} vs y_train {y_train.shape}")
            return False
        
        # Check for NaN/Inf and clean
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            logger.warning("NaN/Inf in training features, cleaning data")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            logger.warning("NaN/Inf in training targets, cleaning data")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Validate validation data if provided
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                logger.error("Validation data shape mismatch")
                return False
                
        return True
    
    def validate_data_distribution(self, X: np.ndarray, y: np.ndarray):
        """Validate data distribution for training"""
        try:
            # Check for constant features
            constant_features = []
            for i in range(X.shape[1]):
                if np.std(X[:, i]) < 1e-8:
                    constant_features.append(i)
                    
            if constant_features:
                logger.warning(f"Found {len(constant_features)} constant features: {constant_features[:5]}...")
                
            # Check target distribution
            y_std = np.std(y)
            if y_std < 1e-8:
                logger.warning("Target variable has very low variance")
            
            # Check for outliers
            y_mean = np.mean(y)
            outliers = np.abs(y - y_mean) > 5 * y_std
            if outliers.any():
                logger.warning(f"Found {outliers.sum()} outliers in target ({outliers.sum() / len(y) * 100:.2f}%)")
                
        except Exception as e:
            logger.error(f"Error validating data distribution: {e}")
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      random_seeds: List[int] = [42, 123, 456]) -> List:
        """Train LightGBM models with error handling"""
        logger.info("Training LightGBM models...")
        
        models = []
        
        for seed in random_seeds:
            try:
                params = self.lgb_params.copy()
                params['seed'] = seed
                params['feature_fraction_seed'] = seed + 1
                params['bagging_seed'] = seed + 2
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Add callbacks
                callbacks = [
                    lgb.early_stopping(50), 
                    lgb.log_evaluation(0)
                ]
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=callbacks
                )
                
                # Validate model
                val_pred = model.predict(X_val, num_iteration=model.best_iteration)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                logger.info(f"LightGBM model with seed {seed} trained, val RMSE: {val_rmse:.6f}")
                
                models.append(model)
                
            except Exception as e:
                logger.error(f"Failed to train LightGBM with seed {seed}: {e}")
                continue
        
        if models:
            logger.info(f"Successfully trained {len(models)} LightGBM models")
        else:
            logger.error("Failed to train any LightGBM models")
            
        return models
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     random_seeds: List[int] = [42, 123, 456]) -> List:
        """Train XGBoost models with error handling"""
        logger.info("Training XGBoost models...")
        
        models = []
        
        for seed in random_seeds:
            try:
                params = self.xgb_params.copy()
                params['random_state'] = seed
                params['seed'] = seed
                
                model = xgb.XGBRegressor(**params)
                
                # Fit with validation
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                # Validate model
                val_pred = model.predict(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                logger.info(f"XGBoost model with seed {seed} trained, val RMSE: {val_rmse:.6f}")
                
                models.append(model)
                
            except Exception as e:
                logger.error(f"Failed to train XGBoost with seed {seed}: {e}")
                continue
        
        if models:
            logger.info(f"Successfully trained {len(models)} XGBoost models")
        else:
            logger.error("Failed to train any XGBoost models")
            
        return models
