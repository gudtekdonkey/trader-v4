"""
Main Package - Trading bot orchestration system
Contains the core orchestration logic and supporting components for running
the trading bot including health monitoring, ML predictions, and task supervision.

File: __init__.py
Modified: 2025-07-19
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
