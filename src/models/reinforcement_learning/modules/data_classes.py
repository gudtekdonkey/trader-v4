"""
Data classes and types for multi-agent trading system
"""

from dataclasses import dataclass


@dataclass
class RLAction:
    """Data class representing a trading action with metadata."""
    action_type: str  # 'hold', 'buy', 'sell'
    intensity: float  # 0.0 to 1.0
    confidence: float
    expected_reward: float