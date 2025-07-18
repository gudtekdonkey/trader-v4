"""
Model health monitoring module for ensemble predictor.
Tracks model health and performance metrics.
"""

from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelHealthMonitor:
    """Monitor individual model health and performance"""
    
    def __init__(self):
        self.model_stats = defaultdict(lambda: {
            'prediction_count': 0,
            'error_count': 0,
            'nan_count': 0,
            'last_error': None,
            'consecutive_errors': 0,
            'performance_history': []
        })
        
    def record_prediction(self, model_name: str, success: bool, has_nan: bool = False):
        """Record prediction attempt"""
        stats = self.model_stats[model_name]
        stats['prediction_count'] += 1
        
        if not success:
            stats['error_count'] += 1
            stats['consecutive_errors'] += 1
            stats['last_error'] = datetime.now()
        else:
            stats['consecutive_errors'] = 0
            
        if has_nan:
            stats['nan_count'] += 1
    
    def record_performance(self, model_name: str, metric: float):
        """Record model performance metric"""
        stats = self.model_stats[model_name]
        stats['performance_history'].append(metric)
        
        # Keep only recent history
        if len(stats['performance_history']) > 100:
            stats['performance_history'] = stats['performance_history'][-100:]
    
    def is_model_healthy(self, model_name: str, error_threshold: int = 10) -> bool:
        """Check if model is healthy"""
        stats = self.model_stats[model_name]
        return stats['consecutive_errors'] < error_threshold
    
    def get_model_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        report = {}
        for model_name, stats in self.model_stats.items():
            error_rate = stats['error_count'] / stats['prediction_count'] if stats['prediction_count'] > 0 else 0
            nan_rate = stats['nan_count'] / stats['prediction_count'] if stats['prediction_count'] > 0 else 0
            
            report[model_name] = {
                'healthy': self.is_model_healthy(model_name),
                'error_rate': error_rate,
                'nan_rate': nan_rate,
                'consecutive_errors': stats['consecutive_errors'],
                'last_error': stats['last_error'],
                'avg_performance': np.mean(stats['performance_history']) if stats['performance_history'] else None
            }
            
        return report
    
    def reset_model_stats(self, model_name: str):
        """Reset statistics for a specific model"""
        if model_name in self.model_stats:
            self.model_stats[model_name] = {
                'prediction_count': 0,
                'error_count': 0,
                'nan_count': 0,
                'last_error': None,
                'consecutive_errors': 0,
                'performance_history': []
            }
            logger.info(f"Reset statistics for {model_name}")
