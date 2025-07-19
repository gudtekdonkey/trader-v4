"""
Reinforcement Learning module for the cryptocurrency trading system.

This module contains the multi-agent reinforcement learning system and risk-aware
extensions for adaptive trading strategies using SAC (Soft Actor-Critic) agents.

Main Components:
    - MultiAgentTradingSystem: Coordinates multiple RL agents for different aspects of trading
    - SACAgent: Soft Actor-Critic agent implementation
    - TradingEnvironment: Custom trading environment for RL agents
    - RiskAwareExtensions: Risk-aware modifications to standard RL algorithms
    - RewardFunctions: Various reward function implementations

Usage:
    from models.reinforcement_learning import MultiAgentTradingSystem
    
    rl_system = MultiAgentTradingSystem(device='cuda')
    action = await rl_system.get_action(state)

Author: Trading Bot Team
Version: 4.0
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
