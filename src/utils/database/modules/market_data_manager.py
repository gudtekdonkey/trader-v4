"""
Market data storage and retrieval
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Manages market data operations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
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
            
            logger.info(f"Saved {len(df_save)} rows of market data for {symbol}")
    
    def load_market_data(self, symbol: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
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
