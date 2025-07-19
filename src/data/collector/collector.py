"""
Real-Time Data Collector - Hyperliquid DEX data collection coordinator
Manages WebSocket connections, data validation, Redis storage, and callback
systems for real-time orderbook, trade, and funding rate data.

File: collector.py
Modified: 2025-07-19
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import defaultdict, deque

from .modules.connection_manager import ConnectionManager
from .modules.data_processor import DataProcessor
from .modules.data_validator import DataValidator
from .modules.subscription_manager import SubscriptionManager
from .modules.redis_manager import RedisManager
from .modules.api_client import APIClient
from .modules.health_monitor import ConnectionHealth

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DataCollector:
    """Real-time data collection from Hyperliquid DEX with modular architecture"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        # Initialize components
        self.connection_manager = ConnectionManager()
        self.data_processor = DataProcessor()
        self.data_validator = DataValidator()
        self.subscription_manager = SubscriptionManager()
        self.redis_manager = RedisManager(redis_host, redis_port)
        self.api_client = APIClient()
        
        # Data buffers with size limits
        self.orderbook_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.trade_buffer = defaultdict(lambda: deque(maxlen=10000))
        self.funding_buffer = defaultdict(lambda: deque(maxlen=100))
        
        # Callbacks
        self.callbacks = defaultdict(list)
        
        # Connection health tracking
        self.connection_health = defaultdict(ConnectionHealth)
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_dropped': 0,
            'last_update': None,
            'errors': defaultdict(int),
            'data_errors': defaultdict(int)
        }
        
        logger.info("DataCollector initialized with modular architecture")
    
    async def connect(self):
        """Initialize WebSocket connections"""
        await self.connection_manager.connect()
        
        # Set up message handlers
        self.connection_manager.set_message_handler(self._process_message)
        
        logger.info("DataCollector connected")
    
    async def _process_message(self, connection_name: str, data: Dict):
        """Process WebSocket message"""
        try:
            self.stats['messages_received'] += 1
            
            # Process based on message type
            result = await self.data_processor.process_message(connection_name, data)
            
            if result:
                # Update buffers
                if result['type'] == 'orderbook':
                    self.orderbook_buffer[result['symbol']].append(result['data'])
                elif result['type'] == 'trades':
                    for trade in result['data']:
                        self.trade_buffer[result['symbol']].append(trade)
                elif result['type'] == 'funding':
                    self.funding_buffer[result['symbol']].append(result['data'])
                
                # Store in Redis
                await self.redis_manager.store_data(result['type'], result['symbol'], result['data'])
                
                # Trigger callbacks
                await self._trigger_callbacks(result['type'], result['data'])
                
                self.stats['messages_processed'] += 1
                self.stats['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing {connection_name} message: {e}")
            self.stats['errors']['processing'] += 1
    
    async def _trigger_callbacks(self, data_type: str, data: Any):
        """Trigger registered callbacks"""
        callback_type = data_type.rstrip('s')  # Remove plural
        for callback in self.callbacks[callback_type]:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"{callback_type} callback error: {e}")
                self.stats['errors']['callback'] += 1
    
    async def subscribe_orderbook(self, symbols: List[str], callback: Optional[Callable] = None):
        """Subscribe to orderbook updates"""
        if callback:
            self.callbacks['orderbook'].append(callback)
        
        await self.subscription_manager.subscribe(
            self.connection_manager, 
            'orderbook', 
            symbols
        )
    
    async def subscribe_trades(self, symbols: List[str], callback: Optional[Callable] = None):
        """Subscribe to trade updates"""
        if callback:
            self.callbacks['trades'].append(callback)
        
        await self.subscription_manager.subscribe(
            self.connection_manager,
            'trades',
            symbols
        )
    
    async def subscribe_funding(self, symbols: List[str], callback: Optional[Callable] = None):
        """Subscribe to funding rate updates"""
        if callback:
            self.callbacks['funding'].append(callback)
        
        await self.subscription_manager.subscribe(
            self.connection_manager,
            'funding',
            symbols
        )
    
    async def get_historical_data(self, symbol: str, interval: str = '1h', 
                                 limit: int = 1000):
        """Fetch historical OHLCV data"""
        return await self.api_client.get_historical_data(symbol, interval, limit)
    
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get latest orderbook data"""
        # Try Redis first
        data = self.redis_manager.get_latest('orderbook', symbol)
        if data:
            return data
        
        # Fallback to buffer
        if symbol in self.orderbook_buffer and self.orderbook_buffer[symbol]:
            latest = self.orderbook_buffer[symbol][-1].copy()
            if 'timestamp' in latest and isinstance(latest['timestamp'], datetime):
                latest['timestamp'] = latest['timestamp'].isoformat()
            return latest
        
        return None
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        # Try Redis first
        trades = self.redis_manager.get_recent_trades(symbol, limit)
        if trades:
            return trades
        
        # Fallback to buffer
        if symbol in self.trade_buffer:
            trades = list(self.trade_buffer[symbol])[-limit:]
            result = []
            for trade in trades:
                trade_copy = trade.copy()
                if 'timestamp' in trade_copy and isinstance(trade_copy['timestamp'], datetime):
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                result.append(trade_copy)
            return result
        
        return []
    
    def get_connection_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health metrics for all connections"""
        health_report = self.connection_manager.get_health_metrics()
        
        # Add overall statistics
        health_report['statistics'] = {
            'messages_received': self.stats['messages_received'],
            'messages_processed': self.stats['messages_processed'],
            'messages_dropped': self.stats['messages_dropped'],
            'errors': dict(self.stats['errors']),
            'data_errors': dict(self.stats['data_errors']),
            'redis_connected': self.redis_manager.test_connection()
        }
        
        return health_report
    
    async def listen(self):
        """Main listening loop"""
        await self.connection_manager.listen()
    
    async def close(self):
        """Close all connections gracefully"""
        logger.info("Closing data collector...")
        
        await self.connection_manager.close()
        self.redis_manager.close()
        
        logger.info("Data collector closed")
