"""
Data processing for different message types
"""

import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes different types of market data messages"""
    
    def __init__(self):
        self.processors = {
            'l2Book': self._process_orderbook,
            'trades': self._process_trades,
            'funding': self._process_funding
        }
    
    async def process_message(self, connection_name: str, data: Dict) -> Optional[Dict]:
        """Process a message based on its type"""
        try:
            msg_type = data.get('channel', data.get('type'))
            
            if msg_type in self.processors:
                return await self.processors[msg_type](data)
            elif msg_type == 'error':
                logger.error(f"Server error: {data}")
                return None
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    async def _process_orderbook(self, data: Dict) -> Optional[Dict]:
        """Process orderbook data"""
        try:
            if 'data' not in data:
                return None
            
            book_data = data['data']
            symbol = book_data.get('coin')
            if not symbol:
                return None
            
            # Extract orderbook data
            levels = book_data.get('levels', {})
            bids = levels.get('bids', [])
            asks = levels.get('asks', [])
            
            if not bids or not asks:
                logger.warning(f"Empty orderbook for {symbol}")
                return None
            
            # Convert and validate price/size pairs
            try:
                bids = [(float(p), float(s)) for p, s in bids]
                asks = [(float(p), float(s)) for p, s in asks]
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid orderbook data format: {e}")
                return None
            
            # Calculate orderbook metrics
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else float('inf')
            
            if best_bid <= 0 or best_ask <= best_bid:
                logger.warning(f"Invalid quotes for {symbol}: bid={best_bid}, ask={best_ask}")
                return None
            
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate order imbalance
            bid_volume = sum(s for _, s in bids[:10])
            ask_volume = sum(s for _, s in asks[:10])
            total_volume = bid_volume + ask_volume
            
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            orderbook_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'mid_price': mid_price,
                'imbalance': imbalance,
                'bid_levels': len(bids),
                'ask_levels': len(asks),
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }
            
            return {
                'type': 'orderbook',
                'symbol': symbol,
                'data': orderbook_data
            }
            
        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
            return None
    
    async def _process_trades(self, data: Dict) -> Optional[Dict]:
        """Process trade data"""
        try:
            if 'data' not in data:
                return None
            
            trade_data = data['data']
            trades = trade_data.get('trades', [])
            if not trades:
                return None
            
            symbol = trade_data.get('coin')
            if not symbol:
                return None
            
            processed_trades = []
            
            for trade in trades:
                try:
                    # Extract and validate trade data
                    price = float(trade.get('px', 0))
                    size = float(trade.get('sz', 0))
                    side = trade.get('side', '').lower()
                    
                    if price <= 0 or size <= 0:
                        continue
                    
                    if side not in ['buy', 'sell']:
                        continue
                    
                    trade_obj = {
                        'symbol': symbol,
                        'price': price,
                        'size': size,
                        'side': side,
                        'timestamp': datetime.now(),
                        'trade_id': trade.get('tid', '')
                    }
                    
                    processed_trades.append(trade_obj)
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid trade data: {e}")
                    continue
            
            if not processed_trades:
                return None
            
            return {
                'type': 'trades',
                'symbol': symbol,
                'data': processed_trades
            }
            
        except Exception as e:
            logger.error(f"Error processing trades: {e}")
            return None
    
    async def _process_funding(self, data: Dict) -> Optional[Dict]:
        """Process funding rate data"""
        try:
            if 'data' not in data:
                return None
            
            funding_data = data['data']
            symbol = funding_data.get('coin')
            if not symbol:
                return None
            
            try:
                funding_rate = float(funding_data.get('fundingRate', 0))
                next_funding_time = funding_data.get('nextFundingTime')
                
                # Validate funding rate (typically between -0.01 and 0.01)
                if abs(funding_rate) > 0.1:  # 10% sanity check
                    logger.warning(f"Unusual funding rate for {symbol}: {funding_rate}")
                
                funding_obj = {
                    'symbol': symbol,
                    'funding_rate': funding_rate,
                    'next_funding_time': next_funding_time,
                    'timestamp': datetime.now()
                }
                
                return {
                    'type': 'funding',
                    'symbol': symbol,
                    'data': funding_obj
                }
                
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid funding data: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing funding: {e}")
            return None
