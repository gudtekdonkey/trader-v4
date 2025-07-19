"""
Database management for trading bot
"""

import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Optional

from .modules.schema_manager import SchemaManager
from .modules.trade_manager import TradeManager
from .modules.position_manager import PositionManager
from .modules.performance_manager import PerformanceManager
from .modules.market_data_manager import MarketDataManager
from .modules.analytics_manager import AnalyticsManager

from ..logger import setup_logger

logger = setup_logger(__name__)


class DatabaseManager:
    """Database management for trading bot with modular architecture"""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        schema_manager = SchemaManager(self.db_path)
        schema_manager.initialize_database()
        
        # Initialize managers
        self.trade_manager = TradeManager(self.db_path)
        self.position_manager = PositionManager(self.db_path)
        self.performance_manager = PerformanceManager(self.db_path)
        self.market_data_manager = MarketDataManager(self.db_path)
        self.analytics_manager = AnalyticsManager(self.db_path)
        
        logger.info("DatabaseManager initialized with modular architecture")
    
    # Trade operations
    def record_trade(self, trade: Dict) -> int:
        """Record a trade in the database"""
        return self.trade_manager.record_trade(trade)
    
    def update_trade(self, trade_id: int, updates: Dict):
        """Update trade record"""
        self.trade_manager.update_trade(trade_id, updates)
    
    def get_trade_history(self, symbol: Optional[str] = None, days: int = 30):
        """Get trade history"""
        return self.trade_manager.get_trade_history(symbol, days)
    
    # Position operations
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return self.position_manager.get_open_positions()
    
    def update_position(self, symbol: str, updates: Dict):
        """Update or insert position"""
        self.position_manager.update_position(symbol, updates)
    
    def remove_position(self, symbol: str):
        """Remove position from database"""
        self.position_manager.remove_position(symbol)
    
    # Performance operations
    def record_performance_metric(self, metric_name: str, metric_value: float, 
                                timeframe: str = None, strategy: str = None):
        """Record performance metric"""
        self.performance_manager.record_metric(metric_name, metric_value, timeframe, strategy)
    
    def record_prediction(self, prediction: Dict):
        """Record model prediction"""
        self.performance_manager.record_prediction(prediction)
    
    def update_prediction_actual(self, prediction_id: int, actual_value: float):
        """Update prediction with actual value"""
        self.performance_manager.update_prediction_actual(prediction_id, actual_value)
    
    def record_risk_event(self, event_type: str, severity: str, 
                         description: str, action_taken: str = None):
        """Record risk management event"""
        self.performance_manager.record_risk_event(event_type, severity, description, action_taken)
    
    # Market data operations
    def save_market_data(self, df, symbol: str):
        """Save market data for backtesting"""
        self.market_data_manager.save_market_data(df, symbol)
    
    def load_market_data(self, symbol: str, start_date=None, end_date=None):
        """Load market data from database"""
        return self.market_data_manager.load_market_data(symbol, start_date, end_date)
    
    # Analytics operations
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary"""
        return self.analytics_manager.get_performance_summary(days)
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data"""
        self.analytics_manager.cleanup_old_data(days_to_keep)
