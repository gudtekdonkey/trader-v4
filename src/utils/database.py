"""
Database Manager Wrapper - Backward compatibility module
Provides a compatibility layer for the refactored database module to maintain
existing import paths.

File: database.py
Modified: 2025-07-19
"""

from .database.database import DatabaseManager

# Export DatabaseManager at the module level for backward compatibility
__all__ = ['DatabaseManager']
