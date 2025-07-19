"""
Performance metrics and predictions management
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class PerformanceManager:
    """Manages performance metrics and prediction operations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    def record_metric(self, metric_name: str, metric_value: float, 
                     timeframe: str = None, strategy: str = None):
        """Record performance metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_metrics (metric_name, metric_value, timeframe, strategy)
                VALUES (?, ?, ?, ?)
            """, (metric_name, metric_value, timeframe, strategy))
            
            conn.commit()
            logger.debug(f"Recorded metric {metric_name}: {metric_value}")
    
    def record_prediction(self, prediction: Dict):
        """Record model prediction"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_predictions 
                (symbol, model_name, prediction_type, prediction_value, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction['symbol'],
                prediction['model_name'],
                prediction['prediction_type'],
                prediction['prediction_value'],
                prediction.get('confidence'),
                json.dumps(prediction.get('metadata', {}))
            ))
            
            conn.commit()
            logger.debug(f"Recorded prediction for {prediction['symbol']}")
    
    def update_prediction_actual(self, prediction_id: int, actual_value: float):
        """Update prediction with actual value"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE model_predictions 
                SET actual_value = ?
                WHERE id = ?
            """, (actual_value, prediction_id))
            
            conn.commit()
    
    def record_risk_event(self, event_type: str, severity: str, 
                         description: str, action_taken: str = None):
        """Record risk management event"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO risk_events (event_type, severity, description, action_taken)
                VALUES (?, ?, ?, ?)
            """, (event_type, severity, description, action_taken))
            
            conn.commit()
            
            logger.warning(f"Risk event recorded: {event_type} - {description}")
