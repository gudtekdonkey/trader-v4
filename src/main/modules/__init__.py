"""
Main bot module components.

This submodule contains the core components for the trading bot orchestration.

Components:
    - HealthMonitor: Component health tracking and monitoring
    - ComponentInitializer: Component initialization with dependency management
    - ConfigValidator: Configuration validation and default values
    - MLPredictor: Machine learning model predictions
    - TaskSupervisor: Task supervision and shutdown management

Author: Trading Bot Team
Version: 4.0
"""

from .health_monitoring import HealthMonitor
from .component_initializer import ComponentInitializer
from .config_validator import ConfigValidator
from .ml_predictor import MLPredictor
from .task_supervisor import TaskSupervisor

__all__ = [
    'HealthMonitor',
    'ComponentInitializer',
    'ConfigValidator',
    'MLPredictor',
    'TaskSupervisor'
]
