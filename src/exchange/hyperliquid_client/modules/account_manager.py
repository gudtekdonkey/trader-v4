"""Account management operations for Hyperliquid"""

import aiohttp
from typing import Dict, List
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class AccountManager:
    """Handles account operations for Hyperliquid"""
    
    def __init__(self, api_url: str, user_address: str):
        self.api_url = api_url
        self.user_address = user_address
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        endpoint = f"{self.api_url}/info"
        payload = {
            "type": "clearinghouseState",
            "req": {"user": self.user_address}
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
            "req": {"user": self.user_address}
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
