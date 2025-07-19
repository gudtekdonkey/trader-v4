"""
Reinforcement Learning module components.

This submodule contains the core components for the multi-agent RL system.

Components:
    - SACAgent: Soft Actor-Critic agent implementation
    - TradingEnvironment: Custom trading environment
    - MultiAgentCoordinator: Coordinates multiple agents
    - RewardFunctions: Various reward function implementations
    - ReplayBuffer: Experience replay buffer
    - NetworkDefinitions: Neural network architectures
    - DataClasses: Data structures for RL components

Author: Trading Bot Team
Version: 4.0
"""

from .sac_agent import SACAgent
from .trading_environment import TradingEnvironment
from .multi_agent_coordinator import MultiAgentCoordinator
from .reward_functions import RewardFunctions
from .replay_buffer import ReplayBuffer
from .network_definitions import ActorNetwork, CriticNetwork, ValueNetwork
from .data_classes import Experience, AgentConfig

__all__ = [
    'SACAgent',
    'TradingEnvironment', 
    'MultiAgentCoordinator',
    'RewardFunctions',
    'ReplayBuffer',
    'ActorNetwork',
    'CriticNetwork', 
    'ValueNetwork',
    'Experience',
    'AgentConfig'
]
