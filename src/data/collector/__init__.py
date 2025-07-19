"""
Data Collector Package - Real-time data collection modules
Provides WebSocket-based data collection from Hyperliquid DEX with
modular architecture for connection management and data processing.

File: __init__.py
Modified: 2025-07-19
"""

from .collector import DataCollector

__all__ = ['DataCollector']
