"""
Database management for trading bot - backward compatibility wrapper
"""

from .database.database import DatabaseManager

# Export DatabaseManager at the module level for backward compatibility
__all__ = ['DatabaseManager']
