"""
Reinforcement Learning Package - Multi-agent RL trading system
Contains multi-agent reinforcement learning system with SAC agents for
adaptive cryptocurrency trading strategies with risk-aware extensions.

File: __init__.py
Modified: 2025-07-19
"""

from typing import TYPE_CHECKING

from .multi_agent_system import MultiAgentTradingSystem
from .risk_aware_extensions import RiskAwareExtensions

# Type checking imports
if TYPE_CHECKING:
    from .modules.sac_agent import SACAgent
    from .modules.trading_environment import TradingEnvironment
    from .modules.multi_agent_coordinator import MultiAgentCoordinator
    from .modules.reward_functions import RewardFunctions

__all__ = ['MultiAgentTradingSystem', 'RiskAwareExtensions']

# Version info
__version__ = '4.0.0'
__author__ = 'Trading Bot Team'
