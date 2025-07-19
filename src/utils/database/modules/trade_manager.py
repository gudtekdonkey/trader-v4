"""
Trade operations management
"""

import sqlite3
import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TradeManager:
    """Manages trade-related database operations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    def record_trade(self, trade: Dict) -> int:
        """Record a trade in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (symbol, side, size, entry_price, exit_price, 
                                  pnl, fees, strategy, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['symbol'],
                trade['side'],
                trade['size'],
                trade['entry_price'],
                trade.get('exit_price'),
                trade.get('pnl'),
                trade.get('fees', 0),
                trade.get('strategy'),
                trade.get('status', 'open'),
                json.dumps(trade.get('metadata', {}))
            ))
            
            trade_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Recorded trade {trade_id}: {trade['symbol']} {trade['side']}")
            
            return trade_id
    
    def update_trade(self, trade_id: int, updates: Dict):
        """Update trade record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert metadata to JSON if present
            if 'metadata' in updates:
                updates['metadata'] = json.dumps(updates['metadata'])
            
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [trade_id]
            
            cursor.execute(f"""
                UPDATE trades 
                SET {set_clause}
                WHERE id = ?
            """, values)
            
            conn.commit()
            logger.info(f"Updated trade {trade_id}")
    
    def get_trade_history(self, symbol: Optional[str] = None, 
                         days: int = 30) -> pd.DataFrame:
        """Get trade history"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM trades 
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days)
            
            if symbol:
                query += f" AND symbol = '{symbol}'"
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Parse metadata JSON
                df['metadata'] = df['metadata'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df
