"""Telegram API communication handler"""

import os
import asyncio
import aiohttp
from typing import Optional
from dataclasses import dataclass
from ...logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for Telegram notifications"""
    bot_token: str
    chat_id: str
    enabled: bool = True
    silent_mode: bool = False
    max_retries: int = 3
    retry_delay: int = 5
    
    @classmethod
    def from_env(cls) -> 'NotificationConfig':
        """Create config from environment variables"""
        return cls(
            bot_token=os.getenv('TELEGRAM_TOKEN', ''),
            chat_id=os.getenv('TELEGRAM_CHAT_ID', ''),
            enabled=os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true',
            silent_mode=os.getenv('TELEGRAM_SILENT', 'false').lower() == 'true'
        )


class TelegramSender:
    """Handles sending messages to Telegram"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{self.config.bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
        if not self.config.bot_token or not self.config.chat_id:
            logger.warning("Telegram credentials not configured. Notifications disabled.")
            self.config.enabled = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def send_message(self, text: str, parse_mode: str = 'HTML', 
                          disable_notification: bool = False) -> bool:
        """Send message to Telegram"""
        if not self.config.enabled:
            return False
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.config.chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_notification': disable_notification or self.config.silent_mode
        }
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(url, json=data) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {error_text}")
            except Exception as e:
                logger.error(f"Error sending Telegram message: {e}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay)
        
        return False
    
    async def test_connection(self) -> bool:
        """Test Telegram connection and credentials"""
        from datetime import datetime, timezone
        
        test_message = f"""
✅ <b>Telegram Notification Test</b>

Your Telegram notifications are configured correctly!
Bot is ready to send trading alerts.

<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        success = await self.send_message(test_message)
        if success:
            logger.info("Telegram connection test successful")
        else:
            logger.error("Telegram connection test failed")
        
        return success
