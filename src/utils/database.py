import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime, timedelta
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    """Database management for trading bot"""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    pnl REAL,
                    fees REAL,
                    strategy TEXT,
                    status TEXT,
                    metadata TEXT
                )
            """)
            
            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    entry_time DATETIME,
                    metadata TEXT
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timeframe TEXT,
                    strategy TEXT
                )
            """)
            
            # Model predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    prediction_value REAL NOT NULL,
                    confidence REAL,
                    actual_value REAL,
                    metadata TEXT
                )
            """)
            
            # Risk events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    action_taken TEXT,
                    metadata TEXT
                )
            """)
            
            # Market data table (for backtesting)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(timestamp, symbol)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
            
            conn.commit()
    
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
            
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [trade_id]
            
            cursor.execute(f"""
                UPDATE trades 
                SET {set_clause}
                WHERE id = ?
            """, values)
            
            conn.commit()
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM positions
            """)
            
            columns = [desc[0] for desc in cursor.description]
            positions = []
            
            for row in cursor.fetchall():
                position = dict(zip(columns, row))
                if position['metadata']:
                    position['metadata'] = json.loads(position['metadata'])
                positions.append(position)
            
            return positions
    
    def update_position(self, symbol: str, updates: Dict):
        """Update or insert position"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if position exists
            cursor.execute("SELECT id FROM positions WHERE symbol = ?", (symbol,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing position
                set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                values = list(updates.values()) + [symbol]
                
                cursor.execute(f"""
                    UPDATE positions 
                    SET {set_clause}
                    WHERE symbol = ?
                """, values)
            else:
                # Insert new position
                updates['symbol'] = symbol
                columns = ", ".join(updates.keys())
                placeholders = ", ".join(["?" for _ in updates])
                
                cursor.execute(f"""
                    INSERT INTO positions ({columns})
                    VALUES ({placeholders})
                """, list(updates.values()))
            
            conn.commit()
    
    def remove_position(self, symbol: str):
        """Remove position from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()
    
    def record_performance_metric(self, metric_name: str, metric_value: float, 
                                timeframe: str = None, strategy: str = None):
        """Record performance metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_metrics (metric_name, metric_value, timeframe, strategy)
                VALUES (?, ?, ?, ?)
            """, (metric_name, metric_value, timeframe, strategy))
            
            conn.commit()
    
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
    
    def save_market_data(self, df: pd.DataFrame, symbol: str):
        """Save market data for backtesting"""
        with sqlite3.connect(self.db_path) as conn:
            # Prepare dataframe
            df_save = df.copy()
            df_save['symbol'] = symbol
            df_save['timestamp'] = df_save.index
            
            # Save to database
            df_save[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                'market_data', 
                conn, 
                if_exists='append', 
                index=False
            )
    
    def load_market_data(self, symbol: str, start_date: datetime = None, 
                        end_date: datetime = None) -> pd.DataFrame:
        """Load market data from database"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
    
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
                
            return df
    
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
                trade_stats['win_rate'] = trade_stats['winning_trades'] / trade_stats['total_trades']
                trade_stats['profit_factor'] = abs(trade_stats['total_pnl'] / trade_stats['max_loss']) if trade_stats['max_loss'] < 0 else float('inf')
            
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
            
            # Clean up old metrics
            cursor.execute("""
                DELETE FROM performance_metrics 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            # Clean up old predictions
            cursor.execute("""
                DELETE FROM model_predictions 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            conn.commit()
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
