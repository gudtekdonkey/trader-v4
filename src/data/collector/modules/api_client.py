"""
API client for historical data
"""

import aiohttp
import pandas as pd
import logging
import time
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class APIClient:
    """Client for REST API interactions"""
    
    def __init__(self):
        self.api_url = "https://api.hyperliquid.xyz"
        
        # Cache settings
        self.cache_ttl = 300  # 5 minutes
        self.cache = {}
        self.cache_timestamps = {}
    
    async def get_historical_data(self, symbol: str, interval: str = '1h', 
                                 limit: int = 1000) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            # Validate inputs
            valid_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if interval not in valid_intervals:
                raise ValueError(f"Invalid interval: {interval}. Must be one of {valid_intervals}")
            
            if not 1 <= limit <= 5000:
                logger.warning(f"Limit {limit} out of range, clamping to [1, 5000]")
                limit = max(1, min(limit, 5000))
            
            # Check cache
            cache_key = f"historical:{symbol}:{interval}:{limit}"
            if cache_key in self.cache:
                cache_time = self.cache_timestamps.get(cache_key, 0)
                if time.time() - cache_time < self.cache_ttl:
                    logger.debug(f"Returning cached historical data for {symbol}")
                    return self.cache[cache_key].copy()
            
            endpoint = f"{self.api_url}/info"
            
            # Calculate time range
            now = datetime.now()
            interval_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }
            
            minutes_back = limit * interval_minutes[interval]
            start_time = now - timedelta(minutes=minutes_back)
            
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": interval,
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(now.timestamp() * 1000)
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"API error {response.status}: {error_text}")
                    
                    data = await response.json()
                    
                    if 'candles' not in data or not data['candles']:
                        logger.warning(f"No candle data returned for {symbol}")
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data['candles'])
                    
                    # Set column names
                    expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if len(df.columns) >= len(expected_columns):
                        df.columns = expected_columns[:len(df.columns)]
                    
                    # Convert timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Convert to numeric
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove invalid rows
                    df = df.dropna()
                    
                    # Validate OHLC relationships
                    invalid_mask = (
                        (df['high'] < df['low']) |
                        (df['high'] < df['open']) |
                        (df['high'] < df['close']) |
                        (df['low'] > df['open']) |
                        (df['low'] > df['close']) |
                        (df['volume'] < 0)
                    )
                    
                    if invalid_mask.any():
                        logger.warning(f"Removing {invalid_mask.sum()} invalid OHLC rows")
                        df = df[~invalid_mask]
                    
                    # Sort by timestamp
                    df = df.sort_index()
                    
                    # Cache the result
                    self.cache[cache_key] = df
                    self.cache_timestamps[cache_key] = time.time()
                    
                    # Clean old cache entries
                    self._clean_cache()
                    
                    return df
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching historical data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        try:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > self.cache_ttl
            ]
            
            for key in expired_keys:
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
            
            if expired_keys:
                logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("API cache cleared")
