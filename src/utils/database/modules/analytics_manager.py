"""
Analytics and reporting operations
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger(__name__)


class AnalyticsManager:
    """Manages analytics and reporting operations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get trade statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_win,
                    MIN(pnl) as max_loss
                FROM trades
                WHERE timestamp >= datetime('now', '-{} days')
                AND status = 'closed'
            """.format(days))
            
            trade_stats = dict(zip([desc[0] for desc in cursor.description], 
                                 cursor.fetchone()))
            
            # Get performance metrics
            cursor.execute("""
                SELECT metric_name, AVG(metric_value) as avg_value
                FROM performance_metrics
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY metric_name
            """.format(days))
            
            metrics = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Calculate additional metrics
            if trade_stats['total_trades'] > 0:
                trade_stats['win_rate'] = (trade_stats['winning_trades'] / 
                                          trade_stats['total_trades'])
                
                if trade_stats['max_loss'] < 0:
                    trade_stats['profit_factor'] = abs(trade_stats['total_pnl'] / 
                                                     trade_stats['max_loss'])
                else:
                    trade_stats['profit_factor'] = float('inf')
            
            return {
                'trade_statistics': trade_stats,
                'performance_metrics': metrics
            }
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            # Clean up old trades
            cursor.execute("""
                DELETE FROM trades 
                WHERE timestamp < ? AND status = 'closed'
            """, (cutoff_date,))
            
            trades_deleted = cursor.rowcount
            
            # Clean up old metrics
            cursor.execute("""
                DELETE FROM performance_metrics 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            metrics_deleted = cursor.rowcount
            
            # Clean up old predictions
            cursor.execute("""
                DELETE FROM model_predictions 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            predictions_deleted = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"Cleaned up data older than {days_to_keep} days: "
                       f"{trades_deleted} trades, {metrics_deleted} metrics, "
                       f"{predictions_deleted} predictions")
