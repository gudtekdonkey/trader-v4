"""Order management operations for Hyperliquid"""

import time
import json
import aiohttp
from typing import Dict, List, Optional, Any
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class OrderManager:
    """Handles order operations for Hyperliquid"""
    
    def __init__(self, api_url: str, signer):
        self.api_url = api_url
        self.signer = signer
    
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = 'limit', price: Optional[float] = None,
                         time_in_force: str = 'GTC', post_only: bool = False,
                         reduce_only: bool = False, asset_info: Dict = None,
                         ticker_data: Dict = None) -> Dict:
        """Place an order on Hyperliquid"""
        
        if not asset_info:
            return {'status': 'error', 'error': 'Asset info required'}
        
        asset_id = asset_info['assetId']
        
        # Prepare order
        is_buy = side.lower() == 'buy'
        
        if order_type == 'market':
            # Market orders use limit orders with slippage
            if not ticker_data:
                return {'status': 'error', 'error': 'Ticker data required for market orders'}
                
            if is_buy:
                price = ticker_data['ask'] * 1.01  # 1% slippage
            else:
                price = ticker_data['bid'] * 0.99
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
        signature = self.signer(order_request, timestamp)
        
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
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None,
                          asset_info: Optional[Dict] = None) -> Dict:
        """Cancel an order"""
        # Get asset info if symbol provided
        asset_id = None
        if asset_info:
            asset_id = asset_info['assetId']
        
        cancel_request = {
            "oid": order_id
        }
        
        if asset_id:
            cancel_request["a"] = asset_id
        
        timestamp = int(time.time() * 1000)
        signature = self.signer(cancel_request, timestamp)
        
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
    
    async def get_order_status(self, order_id: str, user_address: str) -> Dict:
        """Get order status"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "orderStatus",
            "req": {
                "user": user_address,
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
