"""WebSocket operations for Hyperliquid"""

import json
import websockets
from typing import Dict, Callable
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class WebSocketManager:
    """Handles WebSocket operations for Hyperliquid"""
    
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws = None
        self.ws_connected = False
    
    async def connect(self):
        """Connect to WebSocket for real-time data"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.ws_connected = True
            logger.info("Connected to Hyperliquid WebSocket")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def subscribe_trades(self, symbol: str, callback: Callable):
        """Subscribe to trade updates"""
        if not self.ws_connected:
            await self.connect()
        
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": symbol
            }
        }
        
        await self.ws.send(json.dumps(subscription))
        
        # Handle messages
        async for message in self.ws:
            data = json.loads(message)
            if data.get('channel') == 'trades' and data.get('data', {}).get('coin') == symbol:
                await callback(data['data'])
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable):
        """Subscribe to orderbook updates"""
        if not self.ws_connected:
            await self.connect()
        
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": symbol
            }
        }
        
        await self.ws.send(json.dumps(subscription))
        
        # Handle messages
        async for message in self.ws:
            data = json.loads(message)
            if data.get('channel') == 'l2Book' and data.get('data', {}).get('coin') == symbol:
                await callback(data['data'])
    
    async def close(self):
        """Close WebSocket connection"""
        if self.ws and self.ws_connected:
            await self.ws.close()
            self.ws_connected = False
            logger.info("Closed WebSocket connection")
