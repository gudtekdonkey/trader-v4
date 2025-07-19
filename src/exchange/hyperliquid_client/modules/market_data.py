"""Market data operations for Hyperliquid"""

import time
import aiohttp
from typing import Dict, List, Optional, Any
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class MarketDataHandler:
    """Handles market data operations for Hyperliquid"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
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
    
    async def get_asset_info(self, symbol: str) -> Optional[Dict]:
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
