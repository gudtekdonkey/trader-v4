"""Hyperliquid DEX client implementation - Main coordinator"""

import asyncio
import time
from typing import Dict, List, Optional, Any

from .modules.auth_manager import AuthManager
from .modules.market_data import MarketDataHandler
from .modules.order_manager import OrderManager
from .modules.account_manager import AccountManager
from .modules.websocket_manager import WebSocketManager

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class HyperliquidClient:
    """Hyperliquid DEX client implementation - Main coordinator"""
    
    def __init__(self, private_key: str, testnet: bool = False):
        # API endpoints
        if testnet:
            self.api_url = "https://api.hyperliquid-testnet.xyz"
            self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws"
        else:
            self.api_url = "https://api.hyperliquid.xyz"
            self.ws_url = "wss://api.hyperliquid.xyz/ws"
        
        # Initialize managers
        self.auth_manager = AuthManager(private_key)
        self.market_data = MarketDataHandler(self.api_url)
        self.order_manager = OrderManager(self.api_url, self.auth_manager.sign_request)
        self.account_manager = AccountManager(self.api_url, self.auth_manager.get_address())
        self.websocket_manager = WebSocketManager(self.ws_url)
        
        # Convenience properties
        self.address = self.auth_manager.get_address()
        
        # Request tracking
        self.nonce = int(time.time() * 1000)
        
        logger.info(f"HyperliquidClient initialized for {'testnet' if testnet else 'mainnet'}")
    
    # Market Data Methods (delegated to MarketDataHandler)
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data"""
        return await self.market_data.get_ticker(symbol)
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """Get order book data"""
        return await self.market_data.get_orderbook(symbol, depth)
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate"""
        return await self.market_data.get_funding_rate(symbol)
    
    async def get_historical_funding(self, symbol: str, start_time: int, end_time: int) -> List[Dict]:
        """Get historical funding rates"""
        return await self.market_data.get_historical_funding(symbol, start_time, end_time)
    
    # Order Methods (delegated to OrderManager)
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = 'limit', price: Optional[float] = None,
                         time_in_force: str = 'GTC', post_only: bool = False,
                         reduce_only: bool = False) -> Dict:
        """Place an order on Hyperliquid"""
        # Get asset info
        asset_info = await self.market_data.get_asset_info(symbol)
        if not asset_info:
            return {'status': 'error', 'error': 'Invalid symbol'}
        
        # Get ticker data for market orders
        ticker_data = None
        if order_type == 'market':
            ticker_data = await self.get_ticker(symbol)
        
        return await self.order_manager.place_order(
            symbol=symbol,
            side=side,
            size=size,
            order_type=order_type,
            price=price,
            time_in_force=time_in_force,
            post_only=post_only,
            reduce_only=reduce_only,
            asset_info=asset_info,
            ticker_data=ticker_data
        )
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """Cancel an order"""
        asset_info = None
        if symbol:
            asset_info = await self.market_data.get_asset_info(symbol)
        
        return await self.order_manager.cancel_order(order_id, symbol, asset_info)
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        return await self.order_manager.get_order_status(order_id, self.address)
    
    # Account Methods (delegated to AccountManager)
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        return await self.account_manager.get_positions()
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        return await self.account_manager.get_account_info()
    
    # WebSocket Methods (delegated to WebSocketManager)
    async def connect_websocket(self):
        """Connect to WebSocket for real-time data"""
        await self.websocket_manager.connect()
    
    async def subscribe_trades(self, symbol: str, callback):
        """Subscribe to trade updates"""
        await self.websocket_manager.subscribe_trades(symbol, callback)
    
    async def subscribe_orderbook(self, symbol: str, callback):
        """Subscribe to orderbook updates"""
        await self.websocket_manager.subscribe_orderbook(symbol, callback)
    
    async def close(self):
        """Close all connections"""
        await self.websocket_manager.close()
    
    # Private helper methods
    async def _get_asset_info(self, symbol: str) -> Optional[Dict]:
        """Get asset information"""
        return await self.market_data.get_asset_info(symbol)
