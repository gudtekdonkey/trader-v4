"""
WebSocket connection management with automatic reconnection
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Optional, Callable, Any
from datetime import datetime
from collections import defaultdict

from .health_monitor import ConnectionHealth

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections with automatic reconnection"""
    
    def __init__(self):
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.ws_connections = {}
        self.subscriptions = defaultdict(set)
        self.connection_health = defaultdict(ConnectionHealth)
        
        # Reconnection settings
        self.max_reconnect_attempts = 10
        self.reconnect_delay_base = 1  # seconds
        self.reconnect_delay_max = 60  # seconds
        
        # Message handler
        self.message_handler: Optional[Callable] = None
        
        # Connection tasks
        self.connection_tasks = []
    
    def set_message_handler(self, handler: Callable):
        """Set the message processing handler"""
        self.message_handler = handler
    
    async def connect(self):
        """Initialize WebSocket connections"""
        connection_tasks = []
        
        for conn_name in ['main', 'orderbook']:
            task = asyncio.create_task(self._maintain_connection(conn_name))
            connection_tasks.append(task)
            self.connection_tasks.append(task)
        
        # Wait for initial connections
        await asyncio.sleep(2)
        
        # Check if at least one connection is established
        if not any(conn_name in self.ws_connections for conn_name in ['main', 'orderbook']):
            raise ConnectionError("Failed to establish any WebSocket connections")
        
        logger.info("WebSocket connection manager started")
    
    async def _maintain_connection(self, name: str):
        """Maintain a WebSocket connection with automatic reconnection"""
        while True:
            try:
                async with self._websocket_connection(name) as ws:
                    self.ws_connections[name] = ws
                    
                    # Re-subscribe to previous subscriptions
                    await self._resubscribe(name)
                    
                    # Listen for messages
                    await self._listen_websocket(name, ws)
                    
            except Exception as e:
                logger.error(f"Error maintaining {name} connection: {e}")
                
                # Remove failed connection
                self.ws_connections.pop(name, None)
                
                # Wait before retry
                await asyncio.sleep(5)
    
    async def _websocket_connection(self, name: str):
        """Context manager for WebSocket connections"""
        connection = None
        reconnect_attempts = 0
        
        while reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Calculate backoff delay
                if reconnect_attempts > 0:
                    delay = min(
                        self.reconnect_delay_base * (2 ** reconnect_attempts),
                        self.reconnect_delay_max
                    )
                    logger.info(f"Reconnecting {name} in {delay}s (attempt {reconnect_attempts + 1})")
                    await asyncio.sleep(delay)
                
                # Connect
                connection = await websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=10 * 1024 * 1024  # 10MB max message size
                )
                
                self.connection_health[name] = ConnectionHealth()
                logger.info(f"WebSocket connection {name} established")
                
                return connection
                
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket {name} connection closed: {e}")
                self.connection_health[name].record_error()
                reconnect_attempts += 1
                
            except Exception as e:
                logger.error(f"WebSocket {name} error: {e}")
                self.connection_health[name].record_error()
                reconnect_attempts += 1
                
            finally:
                if connection and not connection.closed:
                    await connection.close()
        
        raise ConnectionError(f"Max reconnection attempts reached for {name}")
    
    async def _listen_websocket(self, name: str, ws):
        """Listen to WebSocket messages"""
        try:
            async for message in ws:
                try:
                    self.connection_health[name].record_message()
                    
                    # Parse message
                    data = json.loads(message)
                    
                    # Process message through handler
                    if self.message_handler:
                        await self.message_handler(name, data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message on {name}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error processing message on {name}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket {name} connection closed normally")
        except Exception as e:
            logger.error(f"Error in WebSocket {name} listener: {e}")
            raise
    
    async def _resubscribe(self, connection_name: str):
        """Re-subscribe to previous subscriptions after reconnection"""
        try:
            if connection_name not in self.ws_connections:
                return
            
            # Get stored subscriptions for this connection
            for sub_type, symbols in self.subscriptions.items():
                if not symbols:
                    continue
                
                # Determine which connection handles this subscription
                if (connection_name == 'orderbook' and sub_type == 'l2Book') or \
                   (connection_name == 'main' and sub_type in ['trades', 'funding']):
                    
                    subscription = {
                        "method": "subscribe",
                        "subscription": {
                            "type": sub_type,
                            "coins": list(symbols)
                        }
                    }
                    
                    await self.ws_connections[connection_name].send(json.dumps(subscription))
                    logger.info(f"Re-subscribed to {sub_type} on {connection_name}")
                    
        except Exception as e:
            logger.error(f"Error resubscribing on {connection_name}: {e}")
    
    async def send_subscription(self, connection_name: str, subscription_type: str, 
                               symbols: List[str]):
        """Send subscription request"""
        if connection_name not in self.ws_connections:
            logger.warning(f"{connection_name} connection not available")
            return False
        
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": subscription_type,
                "coins": symbols
            }
        }
        
        try:
            await self.ws_connections[connection_name].send(json.dumps(subscription))
            
            # Store subscription for reconnection
            self.subscriptions[subscription_type].update(symbols)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send subscription: {e}")
            return False
    
    def get_health_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get health metrics for all connections"""
        health_report = {}
        
        for conn_name, health in self.connection_health.items():
            health_report[conn_name] = health.get_metrics()
        
        return health_report
    
    async def listen(self):
        """Main listening loop"""
        try:
            while True:
                await asyncio.sleep(60)  # Health check interval
                
                # Log health status
                health = self.get_health_metrics()
                logger.debug(f"Connection health: {health}")
                
        except asyncio.CancelledError:
            logger.info("Listen loop cancelled")
            raise
    
    async def close(self):
        """Close all connections gracefully"""
        logger.info("Closing connection manager...")
        
        # Cancel connection tasks
        for task in self.connection_tasks:
            task.cancel()
        
        if self.connection_tasks:
            await asyncio.gather(*self.connection_tasks, return_exceptions=True)
        
        # Close WebSocket connections
        for name, ws in list(self.ws_connections.items()):
            try:
                if ws and not ws.closed:
                    await ws.close()
                    logger.info(f"Closed {name} WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing {name} connection: {e}")
            finally:
                self.ws_connections.pop(name, None)
        
        logger.info("Connection manager closed")
