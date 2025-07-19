"""
Ensemble model package for cryptocurrency trading.

This package provides an ensemble learning framework that combines multiple
machine learning models to improve prediction accuracy and robustness.

Main components:
    - EnsembleModel: Main ensemble coordination and prediction
    - VersionManager: Model versioning and compatibility management
    - HealthMonitor: Model health tracking and monitoring
    - WeightNetwork: Adaptive weight allocation network
    - ModelTrainer: Training logic for different model types

Example usage:
    from models.ensemble import EnsembleModel
    
    ensemble = EnsembleModel(config)
    predictions = ensemble.predict(features)
"""

from .ensemble import EnsembleModel

__all__ = ['EnsembleModel']

__version__ = '1.0.0'
__author__ = 'Crypto Trading Bot Team'
