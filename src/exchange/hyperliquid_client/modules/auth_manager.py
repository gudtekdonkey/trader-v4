"""Authentication and signing utilities for Hyperliquid"""

import json
import hashlib
from eth_account import Account
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class AuthManager:
    """Handles authentication and request signing"""
    
    def __init__(self, private_key: str):
        self.private_key = private_key
        self.account = Account.from_key(private_key)
        self.address = self.account.address
    
    def sign_request(self, request: Dict, timestamp: int) -> str:
        """Sign request for authentication"""
        # Hyperliquid uses EIP-712 signing
        # This is a simplified version - actual implementation would use proper EIP-712
        message = json.dumps(request, separators=(',', ':'), sort_keys=True)
        message_hash = hashlib.sha256(message.encode()).digest()
        
        # Sign with private key
        signature = self.account.signHash(message_hash)
        
        return signature.signature.hex()
    
    def get_address(self) -> str:
        """Get the account address"""
        return self.address
