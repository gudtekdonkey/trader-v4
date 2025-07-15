import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
import websockets
import aiohttp
import redis
from collections import defaultdict
import logging
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataCollector:
    """Real-time data collection from Hyperliquid DEX"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.api_url = "https://api.hyperliquid.xyz"
        
        # Redis for real-time data storage
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # WebSocket connections pool
        self.ws_connections = {}
        self.subscriptions = defaultdict(set)
        
        # Data buffers
        self.orderbook_buffer = defaultdict(list)
        self.trade_buffer = defaultdict(list)
        self.funding_buffer = defaultdict(list)
        
        # Callbacks
        self.callbacks = defaultdict(list)
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'last_update': None,
            'errors': 0
        }
        
    async def connect(self):
        """Initialize WebSocket connections"""
        try:
            # Main data feed connection
            self.ws_connections['main'] = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            # Orderbook connection
            self.ws_connections['orderbook'] = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            logger.info("WebSocket connections established")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    async def subscribe_orderbook(self, symbols: List[str], callback: Optional[Callable] = None):
        """Subscribe to orderbook updates"""
        if callback:
            self.callbacks['orderbook'].append(callback)
        
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coins": symbols
            }
        }
        
        await self.ws_connections['orderbook'].send(json.dumps(subscription))
        self.subscriptions['orderbook'].update(symbols)
        logger.info(f"Subscribed to orderbook for {symbols}")
    
    async def subscribe_trades(self, symbols: List[str], callback: Optional[Callable] = None):
        """Subscribe to trade updates"""
        if callback:
            self.callbacks['trades'].append(callback)
        
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coins": symbols
            }
        }
        
        await self.ws_connections['main'].send(json.dumps(subscription))
        self.subscriptions['trades'].update(symbols)
        logger.info(f"Subscribed to trades for {symbols}")
    
    async def subscribe_funding(self, symbols: List[str], callback: Optional[Callable] = None):
        """Subscribe to funding rate updates"""
        if callback:
            self.callbacks['funding'].append(callback)
        
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "funding",
                "coins": symbols
            }
        }
        
        await self.ws_connections['main'].send(json.dumps(subscription))
        self.subscriptions['funding'].update(symbols)
        logger.info(f"Subscribed to funding for {symbols}")
    
    async def get_historical_data(self, symbol: str, interval: str = '1h', 
                                 limit: int = 1000) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        endpoint = f"{self.api_url}/info"
        
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": int((datetime.now() - timedelta(days=30)).timestamp() * 1000),
                "endTime": int(datetime.now().timestamp() * 1000)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                if 'candles' in data:
                    df = pd.DataFrame(data['candles'])
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Convert to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    return df
                else:
                    logger.error(f"Failed to fetch historical data: {data}")
                    return pd.DataFrame()
    
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get latest orderbook data from Redis"""
        key = f"orderbook:{symbol}:latest"
        data = self.redis_client.hgetall(key)
        
        if data:
            # Convert Redis strings back to appropriate types
            for field in ['best_bid', 'best_ask', 'spread', 'mid_price', 'imbalance', 
                         'bid_volume', 'ask_volume', 'timestamp']:
                if field in data:
                    data[field] = float(data[field])
            
            for field in ['bid_levels', 'ask_levels']:
                if field in data:
                    data[field] = int(data[field])
            
            return data
        return None
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades from Redis"""
        key = f"trades:{symbol}:latest"
        trades = self.redis_client.lrange(key, 0, limit - 1)
        
        return [json.loads(trade) for trade in trades]
    
    async def close(self):
        """Close all connections"""
        for name, ws in self.ws_connections.items():
            if ws and not ws.closed:
                await ws.close()
                logger.info(f"Closed {name} WebSocket connection")
        
        self.redis_client.close()
        logger.info("Closed Redis connection")
