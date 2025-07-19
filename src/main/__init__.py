"""
Main bot orchestration module for the cryptocurrency trading system.

This module contains the core orchestration logic and supporting components
for running the trading bot, including health monitoring, component initialization,
configuration validation, ML predictions, and task supervision.

Main Components:
    - HyperliquidTradingBot: Main orchestration class for the trading system
    - HealthMonitor: System health and component monitoring
    - ComponentInitializer: Component initialization with dependency management
    - ConfigValidator: Configuration validation and defaults
    - MLPredictor: Machine learning model predictions
    - TaskSupervisor: Task supervision and shutdown management

Usage:
    from main import HyperliquidTradingBot
    
    bot = HyperliquidTradingBot("configs/config.yaml")
    await bot.start()

Author: Trading Bot Team
Version: 4.0
"""

from typing import TYPE_CHECKING

from .main import HyperliquidTradingBot

# Type checking imports
if TYPE_CHECKING:
    from .modules.health_monitoring import HealthMonitor
    from .modules.component_initializer import ComponentInitializer
    from .modules.config_validator import ConfigValidator
    from .modules.ml_predictor import MLPredictor
    from .modules.task_supervisor import TaskSupervisor

__all__ = ['HyperliquidTradingBot']

# Version info
__version__ = '4.0.0'
__author__ = 'G/Oasis/P'
