"""
Position management operations
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages position-related database operations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
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
            
            # Convert metadata to JSON if present
            if 'metadata' in updates:
                updates['metadata'] = json.dumps(updates['metadata'])
            
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
            logger.info(f"Updated position for {symbol}")
    
    def remove_position(self, symbol: str):
        """Remove position from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()
            logger.info(f"Removed position for {symbol}")
