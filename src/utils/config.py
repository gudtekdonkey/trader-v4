"""
Configuration Manager - YAML-based configuration management
Manages application configuration loading from YAML files and environment
variables with validation and default values.

File: config.py
Modified: 2025-07-15
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration management"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_env_vars()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'trading': {
                    'initial_capital': 100000,
                    'max_positions': 10,
                    'risk_per_trade': 0.02,
                    'max_drawdown': 0.20,
                    'symbols': ['BTC-USD', 'ETH-USD', 'SOL-USD']
                },
                'exchange': {
                    'name': 'hyperliquid',
                    'testnet': False,
                    'rate_limit': 100
                },
                'strategies': {
                    'momentum': {'enabled': True, 'weight': 0.3},
                    'mean_reversion': {'enabled': True, 'weight': 0.3},
                    'arbitrage': {'enabled': True, 'weight': 0.2},
                    'market_making': {'enabled': True, 'weight': 0.2}
                },
                'ml_models': {
                    'ensemble_enabled': True,
                    'retrain_interval': 24,  # hours
                    'min_data_points': 1000
                },
                'risk_management': {
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.03,
                    'trailing_stop': True,
                    'max_correlation': 0.7
                },
                'execution': {
                    'slippage_tolerance': 0.002,
                    'order_timeout': 30,
                    'use_limit_orders': True
                },
                'monitoring': {
                    'dashboard_port': 8000,
                    'metrics_interval': 60,
                    'alert_channels': ['telegram', 'email']
                },
                'data': {
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                    'postgres_url': 'postgresql://user:pass@localhost/trading',
                    'data_retention_days': 90
                }
            }
    
    def _load_env_vars(self):
        """Load environment variables and override config"""
        # Exchange credentials
        if 'HYPERLIQUID_PRIVATE_KEY' in os.environ:
            self.config['exchange']['private_key'] = os.environ['HYPERLIQUID_PRIVATE_KEY']
        
        # Database URLs
        if 'REDIS_URL' in os.environ:
            self.config['data']['redis_url'] = os.environ['REDIS_URL']
        
        if 'DATABASE_URL' in os.environ:
            self.config['data']['postgres_url'] = os.environ['DATABASE_URL']
        
        # Trading parameters
        if 'INITIAL_CAPITAL' in os.environ:
            self.config['trading']['initial_capital'] = float(os.environ['INITIAL_CAPITAL'])
        
        # Alert credentials
        if 'TELEGRAM_TOKEN' in os.environ:
            self.config['alerts'] = self.config.get('alerts', {})
            self.config['alerts']['telegram_token'] = os.environ['TELEGRAM_TOKEN']
        
        if 'TELEGRAM_CHAT_ID' in os.environ:
            self.config['alerts']['telegram_chat_id'] = os.environ['TELEGRAM_CHAT_ID']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'exchange.private_key',
            'trading.initial_capital',
            'trading.symbols'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"Missing required configuration: {key}")
                return False
        
        return True
