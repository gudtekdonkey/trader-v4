import asyncio
import json
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Any
import aiohttp
import websockets
from eth_account import Account
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class HyperliquidClient:
    """Hyperliquid DEX client implementation"""
    
    def __init__(self, private_key: str, testnet: bool = False):
        self.private_key = private_key
        self.account = Account.from_key(private_key)
        self.address = self.account.address
        
        # API endpoints
        if testnet:
            self.api_url = "https://api.hyperliquid-testnet.xyz"
            self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws"
        else:
            self.api_url = "https://api.hyperliquid.xyz"
            self.ws_url = "wss://api.hyperliquid.xyz/ws"
        
        # WebSocket connection
        self.ws = None
        self.ws_connected = False
        
        # Request tracking
        self.nonce = int(time.time() * 1000)
        
    async def connect_websocket(self):
        """Connect to WebSocket for real-time data"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.ws_connected = True
            logger.info("Connected to Hyperliquid WebSocket")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "meta",
            "req": {"coin": symbol}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                if 'universe' in data:
                    for asset in data['universe']:
                        if asset['name'] == symbol:
                            return {
                                'symbol': symbol,
                                'bid': float(asset.get('bestBid', 0)),
                                'ask': float(asset.get('bestAsk', 0)),
                                'last': float(asset.get('lastPrice', 0)),
                                'volume': float(asset.get('volume24h', 0)),
                                'timestamp': time.time()
                            }
                
                return {}
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """Get order book data"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "l2Book",
            "req": {
                "coin": symbol,
                "nLevels": depth
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                return {
                    'symbol': symbol,
                    'bids': [[float(b['px']), float(b['sz'])] for b in data.get('levels', {}).get('bids', [])],
                    'asks': [[float(a['px']), float(a['sz'])] for a in data.get('levels', {}).get('asks', [])],
                    'timestamp': time.time()
                }
    
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = 'limit', price: Optional[float] = None,
                         time_in_force: str = 'GTC', post_only: bool = False,
                         reduce_only: bool = False) -> Dict:
        """Place an order on Hyperliquid"""
        
        # Get asset info
        asset_info = await self._get_asset_info(symbol)
        if not asset_info:
            return {'status': 'error', 'error': 'Invalid symbol'}
        
        asset_id = asset_info['assetId']
        
        # Prepare order
        is_buy = side.lower() == 'buy'
        
        if order_type == 'market':
            # Market orders use limit orders with slippage
            ticker = await self.get_ticker(symbol)
            if is_buy:
                price = ticker['ask'] * 1.01  # 1% slippage
            else:
                price = ticker['bid'] * 0.99
            limit_px = str(int(price * 1e8))  # Convert to fixed point
            order_type_wire = {"limit": {"tif": "IOC"}}
        else:
            limit_px = str(int(price * 1e8))
            if post_only:
                order_type_wire = {"limit": {"tif": "Alo"}}  # Add liquidity only
            else:
                order_type_wire = {"limit": {"tif": time_in_force}}
        
        # Build order request
        order_request = {
            "a": asset_id,
            "b": is_buy,
            "p": limit_px,
            "s": str(size),
            "r": reduce_only,
            "t": order_type_wire
        }
        
        # Sign and send order
        timestamp = int(time.time() * 1000)
        signature = self._sign_request(order_request, timestamp)
        
        request = {
            "action": {
                "type": "order",
                "orders": [order_request],
                "grouping": "na"
            },
            "nonce": timestamp,
            "signature": signature
        }
        
        endpoint = f"{self.api_url}/exchange"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=request) as response:
                data = await response.json()
                
                if data.get('status') == 'ok':
                    order_result = data['response']['data']['statuses'][0]
                    
                    if 'resting' in order_result:
                        return {
                            'status': 'success',
                            'order_id': order_result['resting']['oid'],
                            'timestamp': timestamp
                        }
                    elif 'filled' in order_result:
                        return {
                            'status': 'success',
                            'order_id': str(timestamp),
                            'filled': True,
                            'fill_price': float(order_result['filled']['avgPx']) / 1e8,
                            'fill_size': float(order_result['filled']['totalSz']),
                            'timestamp': timestamp
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': order_result.get('error', 'Unknown error')
                        }
                else:
                    return {
                        'status': 'error',
                        'error': data.get('response', {}).get('error', 'Unknown error')
                    }
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """Cancel an order"""
        # Get asset info if symbol provided
        asset_id = None
        if symbol:
            asset_info = await self._get_asset_info(symbol)
            if asset_info:
                asset_id = asset_info['assetId']
        
        cancel_request = {
            "oid": order_id
        }
        
        if asset_id:
            cancel_request["a"] = asset_id
        
        timestamp = int(time.time() * 1000)
        signature = self._sign_request(cancel_request, timestamp)
        
        request = {
            "action": {
                "type": "cancel",
                "cancels": [cancel_request]
            },
            "nonce": timestamp,
            "signature": signature
        }
        
        endpoint = f"{self.api_url}/exchange"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=request) as response:
                data = await response.json()
                
                if data.get('status') == 'ok':
                    return {'status': 'success', 'cancelled': order_id}
                else:
                    return {
                        'status': 'error',
                        'error': data.get('response', {}).get('error', 'Unknown error')
                    }
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "orderStatus",
            "req": {
                "user": self.address,
                "oid": order_id
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                if 'order' in data:
                    order = data['order']
                    return {
                        'status': self._map_order_status(order['orderStatus']),
                        'filled_size': float(order.get('filledSz', 0)),
                        'avg_fill_price': float(order.get('avgFillPx', 0)) / 1e8 if order.get('avgFillPx') else 0,
                        'remaining_size': float(order.get('sz', 0)) - float(order.get('filledSz', 0)),
                        'timestamp': order.get('timestamp', 0)
                    }
                
                return {'status': 'not_found'}
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "clearinghouseState",
            "req": {"user": self.address}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                positions = []
                if 'assetPositions' in data:
                    for pos in data['assetPositions']:
                        position = {
                            'symbol': pos['position']['coin'],
                            'size': float(pos['position']['szi']),
                            'entry_price': float(pos['position']['entryPx']) / 1e8,
                            'unrealized_pnl': float(pos['position']['unrealizedPnl']),
                            'realized_pnl': float(pos['position']['realizedPnl']),
                            'margin_used': float(pos['position']['marginUsed']),
                            'liquidation_price': float(pos['position'].get('liquidationPx', 0)) / 1e8
                        }
                        positions.append(position)
                
                return positions
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "clearinghouseState",
            "req": {"user": self.address}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                return {
                    'equity': float(data.get('marginSummary', {}).get('accountValue', 0)),
                    'balance': float(data.get('marginSummary', {}).get('totalNtlPos', 0)),
                    'margin_used': float(data.get('marginSummary', {}).get('totalMarginUsed', 0)),
                    'available_margin': float(data.get('marginSummary', {}).get('availableMargin', 0)),
                    'leverage': float(data.get('marginSummary', {}).get('leverage', 0)),
                    'positions': len(data.get('assetPositions', []))
                }
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "meta",
            "req": {"coin": symbol}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                if 'universe' in data:
                    for asset in data['universe']:
                        if asset['name'] == symbol:
                            return {
                                'symbol': symbol,
                                'funding_rate': float(asset.get('funding', 0)),
                                'next_funding_time': asset.get('nextFundingTime', 0)
                            }
                
                return {}
    
    async def get_historical_funding(self, symbol: str, start_time: int, end_time: int) -> List[Dict]:
        """Get historical funding rates"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "fundingHistory",
            "req": {
                "coin": symbol,
                "startTime": start_time,
                "endTime": end_time
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                funding_history = []
                for entry in data.get('fundingHistory', []):
                    funding_history.append({
                        'timestamp': entry['time'],
                        'funding_rate': float(entry['fundingRate']),
                        'premium': float(entry.get('premium', 0))
                    })
                
                return funding_history
    
    async def _get_asset_info(self, symbol: str) -> Optional[Dict]:
        """Get asset information"""
        endpoint = f"{self.api_url}/info"
        payload = {"type": "meta"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                data = await response.json()
                
                if 'universe' in data:
                    for i, asset in enumerate(data['universe']):
                        if asset['name'] == symbol:
                            return {
                                'symbol': symbol,
                                'assetId': i,
                                'szDecimals': asset.get('szDecimals', 8)
                            }
                
                return None
    
    def _sign_request(self, request: Dict, timestamp: int) -> str:
        """Sign request for authentication"""
        # Hyperliquid uses EIP-712 signing
        # This is a simplified version - actual implementation would use proper EIP-712
        message = json.dumps(request, separators=(',', ':'), sort_keys=True)
        message_hash = hashlib.sha256(message.encode()).digest()
        
        # Sign with private key
        signature = self.account.signHash(message_hash)
        
        return signature.signature.hex()
    
    def _map_order_status(self, status: str) -> str:
        """Map Hyperliquid order status to internal status"""
        status_map = {
            'open': 'open',
            'filled': 'filled',
            'canceled': 'cancelled',
            'rejected': 'rejected',
            'triggered': 'triggered',
            'expired': 'expired'
        }
        
        return status_map.get(status.lower(), 'unknown')
    
    async def subscribe_trades(self, symbol: str, callback):
        """Subscribe to trade updates"""
        if not self.ws_connected:
            await self.connect_websocket()
        
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
    
    async def subscribe_orderbook(self, symbol: str, callback):
        """Subscribe to orderbook updates"""
        if not self.ws_connected:
            await self.connect_websocket()
        
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
        """Close connections"""
        if self.ws and self.ws_connected:
            await self.ws.close()
            self.ws_connected = False
            logger.info("Closed WebSocket connection")
