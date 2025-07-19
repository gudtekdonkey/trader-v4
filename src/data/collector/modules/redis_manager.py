"""
Redis data storage management
"""

import json
import logging
import redis
from typing import Dict, Optional, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class RedisManager:
    """Manages Redis data storage"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.redis_host = host
        self.redis_port = port
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            logger.warning("Running without Redis - data will not be persisted")
        except Exception as e:
            logger.error(f"Unexpected error initializing Redis: {e}")
            self.redis_client = None
    
    def test_connection(self) -> bool:
        """Test Redis connection"""
        if self.redis_client is None:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            return False
    
    async def store_data(self, data_type: str, symbol: str, data: Any):
        """Store data in Redis"""
        if not self.redis_client:
            return
        
        try:
            if data_type == 'orderbook':
                await self._store_orderbook(symbol, data)
            elif data_type == 'trades':
                await self._store_trades(symbol, data)
            elif data_type == 'funding':
                await self._store_funding(symbol, data)
                
        except Exception as e:
            logger.error(f"Failed to store {data_type} data in Redis: {e}")
    
    async def _store_orderbook(self, symbol: str, data: Dict):
        """Store orderbook data"""
        try:
            key = f"orderbook:{symbol}:latest"
            
            # Convert datetime to string
            store_data = {
                k: str(v) if isinstance(v, datetime) else v
                for k, v in data.items()
            }
            
            self.redis_client.hset(key, mapping=store_data)
            self.redis_client.expire(key, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Failed to store orderbook: {e}")
    
    async def _store_trades(self, symbol: str, trades: List[Dict]):
        """Store trade data"""
        try:
            key = f"trades:{symbol}:latest"
            
            # Store as list with limit
            pipe = self.redis_client.pipeline()
            for trade in trades:
                trade_json = json.dumps({
                    k: str(v) if isinstance(v, datetime) else v
                    for k, v in trade.items()
                })
                pipe.lpush(key, trade_json)
            
            pipe.ltrim(key, 0, 999)  # Keep last 1000 trades
            pipe.expire(key, 3600)
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to store trades: {e}")
    
    async def _store_funding(self, symbol: str, data: Dict):
        """Store funding rate data"""
        try:
            key = f"funding:{symbol}:latest"
            
            self.redis_client.hset(key, mapping={
                'funding_rate': str(data['funding_rate']),
                'next_funding_time': str(data['next_funding_time']),
                'timestamp': str(data['timestamp'])
            })
            self.redis_client.expire(key, 3600)
            
        except Exception as e:
            logger.error(f"Failed to store funding: {e}")
    
    def get_latest(self, data_type: str, symbol: str) -> Optional[Dict]:
        """Get latest data from Redis"""
        if not self.redis_client:
            return None
        
        try:
            key = f"{data_type}:{symbol}:latest"
            data = self.redis_client.hgetall(key)
            
            if data:
                # Convert string values back to appropriate types
                for field in ['best_bid', 'best_ask', 'spread', 'mid_price', 
                             'imbalance', 'bid_volume', 'ask_volume', 'funding_rate']:
                    if field in data:
                        data[field] = float(data[field])
                
                for field in ['bid_levels', 'ask_levels']:
                    if field in data:
                        data[field] = int(data[field])
                
                return data
                
        except Exception as e:
            logger.error(f"Failed to get {data_type} from Redis: {e}")
            return None
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades from Redis"""
        if not self.redis_client:
            return []
        
        try:
            key = f"trades:{symbol}:latest"
            trades = self.redis_client.lrange(key, 0, limit - 1)
            return [json.loads(trade) for trade in trades]
            
        except Exception as e:
            logger.error(f"Failed to get trades from Redis: {e}")
            return []
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
