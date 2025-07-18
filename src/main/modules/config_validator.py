"""
Configuration validation module for the trading bot.
Handles configuration loading, validation, and default value setting.
"""

from typing import Dict, Any, List, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfigValidator:
    """Validates and manages trading bot configuration"""
    
    def __init__(self, config):
        self.config = config
        self.required_sections = {
            'exchange': ['private_key', 'testnet'],
            'trading': ['initial_capital', 'symbols'],
            'ml_models': ['device'],
            'data': ['redis_host', 'redis_port'],
            'risk': ['max_position_size', 'max_drawdown']
        }
        
        self.defaults = {
            'exchange.testnet': True,
            'trading.initial_capital': 10000,
            'trading.symbols': ['BTC-USD'],
            'ml_models.device': 'cpu',
            'data.redis_host': 'localhost',
            'data.redis_port': 6379,
            'risk.max_position_size': 0.1,
            'risk.max_drawdown': 0.2,
            'monitoring.memory_growth_threshold_mb': 100,
            'monitoring.memory_cleanup_threshold_mb': 500,
            'monitoring.task_timeout_seconds': 300,
            'monitoring.portfolio_check_interval': 300,
            'dashboard.enabled': True,
            'dashboard.port': 5000,
            'hedging.enabled': True,
            'alerts.log_alerts': True,
            'alerts.email_alerts': False,
            'alerts.slack_alerts': False,
            'ml_models.train_rl_agents': False,
            'ml_models.rl_training_episodes': 1000,
            'ml_models.rl_models_path': 'models/rl_agents/'
        }
    
    def validate(self) -> bool:
        """Validate configuration with schema checking"""
        try:
            # Basic validation
            if not self.config.validate():
                return False
                
            # Schema validation
            self._validate_schema()
            
            # Type validation
            self._validate_types()
            
            # Range validation
            self._validate_ranges()
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_schema(self):
        """Validate configuration schema and set defaults"""
        for section, keys in self.required_sections.items():
            if not self.config.has_section(section):
                raise ValueError(f"Missing required config section: {section}")
                
            for key in keys:
                full_key = f"{section}.{key}"
                if not self.config.has(full_key):
                    # Try to set sensible defaults
                    if full_key in self.defaults:
                        logger.warning(f"Missing config {full_key}, using default: {self.defaults[full_key]}")
                        self.config.set(full_key, self.defaults[full_key])
                    else:
                        raise ValueError(f"Missing required config key: {full_key}")
        
        # Set any other missing defaults
        for key, value in self.defaults.items():
            if not self.config.has(key):
                logger.info(f"Setting default for {key}: {value}")
                self.config.set(key, value)
    
    def _validate_types(self):
        """Validate configuration value types"""
        type_checks = [
            ('trading.initial_capital', (int, float), lambda x: x > 0),
            ('risk.max_position_size', float, lambda x: 0 < x <= 1),
            ('risk.max_drawdown', float, lambda x: 0 < x <= 1),
            ('data.redis_port', int, lambda x: 1 <= x <= 65535),
            ('dashboard.port', int, lambda x: 1024 <= x <= 65535),
            ('monitoring.memory_growth_threshold_mb', (int, float), lambda x: x > 0),
            ('monitoring.memory_cleanup_threshold_mb', (int, float), lambda x: x > 0),
            ('monitoring.task_timeout_seconds', int, lambda x: x > 0),
        ]
        
        for key, expected_type, validator in type_checks:
            if self.config.has(key):
                value = self.config.get(key)
                
                # Check type
                if not isinstance(value, expected_type):
                    raise TypeError(f"{key} must be of type {expected_type}, got {type(value)}")
                
                # Check validator
                if not validator(value):
                    raise ValueError(f"{key} has invalid value: {value}")
    
    def _validate_ranges(self):
        """Validate configuration value ranges"""
        # Validate initial capital
        capital = self.config.get('trading.initial_capital')
        if capital < 100:
            logger.warning(f"Initial capital ({capital}) is very low, this may limit trading capabilities")
        
        # Validate symbols
        symbols = self.config.get('trading.symbols')
        if not symbols or not isinstance(symbols, list):
            raise ValueError("trading.symbols must be a non-empty list")
        
        # Validate risk parameters
        max_position = self.config.get('risk.max_position_size')
        max_drawdown = self.config.get('risk.max_drawdown')
        
        if max_drawdown < 0.05:
            logger.warning(f"Max drawdown ({max_drawdown}) is very conservative, this may limit trading opportunities")
        
        if max_position > 0.5:
            logger.warning(f"Max position size ({max_position}) is aggressive, ensure proper risk management")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration"""
        return {
            'exchange': {
                'testnet': self.config.get('exchange.testnet'),
                'has_private_key': bool(self.config.get('exchange.private_key'))
            },
            'trading': {
                'initial_capital': self.config.get('trading.initial_capital'),
                'symbols': self.config.get('trading.symbols'),
                'num_symbols': len(self.config.get('trading.symbols', []))
            },
            'ml_models': {
                'device': self.config.get('ml_models.device'),
                'train_rl_agents': self.config.get('ml_models.train_rl_agents')
            },
            'risk': {
                'max_position_size': self.config.get('risk.max_position_size'),
                'max_drawdown': self.config.get('risk.max_drawdown')
            },
            'monitoring': {
                'memory_threshold_mb': self.config.get('monitoring.memory_growth_threshold_mb'),
                'task_timeout_seconds': self.config.get('monitoring.task_timeout_seconds')
            },
            'features': {
                'dashboard_enabled': self.config.get('dashboard.enabled'),
                'hedging_enabled': self.config.get('hedging.enabled'),
                'alerts_enabled': any([
                    self.config.get('alerts.log_alerts'),
                    self.config.get('alerts.email_alerts'),
                    self.config.get('alerts.slack_alerts')
                ])
            }
        }
