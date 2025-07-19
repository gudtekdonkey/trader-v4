"""
Database schema management
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages database schema initialization and migrations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    def initialize_database(self):
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
            logger.info("Database schema initialized")
