# Main Module

## Overview
The main module contains the core orchestration logic for the cryptocurrency trading bot. It coordinates all components and manages the trading lifecycle.

## Structure
```
main/
├── __init__.py              # Package initialization
├── main.py                  # Main bot orchestration class
├── README.md               # This file
└── modules/                # Submodules
    ├── __init__.py
    ├── health_monitoring.py      # System health tracking
    ├── component_initializer.py  # Component initialization
    ├── config_validator.py       # Configuration validation
    ├── ml_predictor.py          # ML model predictions
    └── task_supervisor.py       # Task management
```

## Components

### HyperliquidTradingBot
The main orchestration class that:
- Initializes all trading components
- Manages the trading loop
- Handles shutdown gracefully
- Monitors system health

### HealthMonitor
- Tracks component health status
- Detects memory leaks
- Identifies deadlocked tasks
- Manages component dependencies

### ComponentInitializer
- Initializes trading components in correct order
- Handles initialization failures
- Manages component dependencies
- Supports partial initialization cleanup

### ConfigValidator
- Validates configuration files
- Provides default values
- Checks parameter ranges
- Ensures required settings exist

### MLPredictor
- Coordinates ML model predictions
- Integrates ensemble models
- Manages RL agent actions
- Handles prediction errors

### TaskSupervisor
- Manages async task lifecycle
- Monitors task performance
- Handles task failures
- Coordinates shutdown

## Usage

```python
from main import HyperliquidTradingBot

# Create bot instance
bot = HyperliquidTradingBot("configs/config.yaml")

# Start trading
await bot.start()

# Graceful shutdown
await bot.shutdown()
```

## Configuration

The bot requires a YAML configuration file with sections for:
- Exchange credentials
- Trading parameters
- Risk management settings
- ML model configurations
- Strategy parameters

See `configs/config.example.yaml` for a complete example.

## Error Handling

The main module implements comprehensive error handling:
- Component initialization failures
- Runtime exceptions in trading loop
- Connection issues
- Data feed problems
- Model prediction errors

## Monitoring

Health monitoring includes:
- Memory usage tracking
- CPU utilization
- Component status
- Task deadlock detection
- Performance metrics

## Dependencies

- asyncio for async operations
- pandas for data handling
- numpy for numerical operations
- Custom trading modules

## Testing

Run tests with:
```bash
pytest tests/test_main.py
```

## Contributing

When modifying the main module:
1. Maintain backward compatibility
2. Add proper error handling
3. Update documentation
4. Add unit tests
5. Follow the existing code style
