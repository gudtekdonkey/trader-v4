"""
Subscription management
"""

import logging
from typing import List, Set
from collections import defaultdict

from .data_validator import DataValidator

logger = logging.getLogger(__name__)


class SubscriptionManager:
    """Manages WebSocket subscriptions"""
    
    def __init__(self):
        self.subscriptions = defaultdict(set)
        self.validator = DataValidator()
    
    async def subscribe(self, connection_manager, subscription_type: str, 
                       symbols: List[str]):
        """Subscribe to market data"""
        try:
            # Validate symbols
            valid_symbols = [s for s in symbols if self.validator.validate_symbol(s)]
            if not valid_symbols:
                raise ValueError("No valid symbols provided")
            
            if len(valid_symbols) < len(symbols):
                invalid = set(symbols) - set(valid_symbols)
                logger.warning(f"Invalid symbols filtered out: {invalid}")
            
            # Determine connection and message type
            if subscription_type == 'orderbook':
                connection_name = 'orderbook'
                msg_type = 'l2Book'
            else:
                connection_name = 'main'
                msg_type = subscription_type
            
            # Send subscription
            success = await connection_manager.send_subscription(
                connection_name, 
                msg_type, 
                valid_symbols
            )
            
            if success:
                self.subscriptions[subscription_type].update(valid_symbols)
                logger.info(f"Subscribed to {subscription_type} for {valid_symbols}")
            else:
                logger.warning(f"Failed to subscribe to {subscription_type}")
                
        except Exception as e:
            logger.error(f"Failed to subscribe to {subscription_type}: {e}")
            raise
    
    def get_subscriptions(self, subscription_type: str) -> Set[str]:
        """Get current subscriptions for a type"""
        return self.subscriptions.get(subscription_type, set())
    
    def clear_subscriptions(self, subscription_type: str = None):
        """Clear subscriptions"""
        if subscription_type:
            self.subscriptions[subscription_type].clear()
        else:
            self.subscriptions.clear()
