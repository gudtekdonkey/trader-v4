"""
Database Package - Modular database management
Provides database connection management and query execution with connection
pooling and error handling.

File: __init__.py
Modified: 2025-07-19
"""

from .database import DatabaseManager

__all__ = ['DatabaseManager']
