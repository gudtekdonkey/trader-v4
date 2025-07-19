# Trading Module

## Overview
The trading module contains all trading strategies, execution logic, risk management, and portfolio optimization components for the cryptocurrency trading system.

## Structure
```
trading/
├── __init__.py                    # Package initialization
├── README.md                     # This file
├── strategies/                   # Trading strategies
│   ├── momentum/                # Momentum strategy
│   ├── mean_reversion/         # Mean reversion strategy
│   ├── arbitrage/              # Arbitrage strategies
│   └── market_making/          # Market making strategy
├── execution/                    # Order execution
│   └── advanced_executor/      # Advanced execution algorithms
├── risk_manager/                # Risk management
├── position_sizer/              # Position sizing
├── portfolio/                   # Portfolio management
│   ├── analytics/              # Portfolio analytics
│   └── monitor/                # Portfolio monitoring
├── optimization/                # Portfolio optimization
│   ├── black_litterman/        # Black-Litterman model
│   └── hierarchical_risk_parity/ # HRP optimization
├── regime_detector/             # Market regime detection
├── adaptive_strategy_manager/   # Strategy allocation
├── dynamic_hedging/            # Hedging system
└── order_executor/             # Basic order execution
```

## Core Components

### Trading Strategies

#### Momentum Strategy
- Trend following approach
- Multiple timeframe analysis
- Dynamic stop losses
- Volatility-based position sizing

#### Mean Reversion Strategy
- Statistical arbitrage
- Z-score based signals
- Cointegration testing
- Pairs trading support

#### Arbitrage Strategy
- Triangular arbitrage
- Statistical arbitrage
- Funding rate arbitrage
- Cross-exchange opportunities

#### Market Making Strategy
- Dynamic quote generation
- Inventory management
- Adverse selection protection
- Spread optimization

### Risk Management

#### RiskManager
Core risk control system:
- Position limits
- Drawdown controls
- Correlation monitoring
- VaR/CVaR calculations

#### PositionSizer
Intelligent position sizing:
- Kelly criterion
- Fixed fractional
- Volatility-based
- Risk parity

### Order Execution

#### AdvancedExecutor
Sophisticated execution algorithms:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg orders
- Adaptive algorithms

#### OrderExecutor
Basic order management:
- Market orders
- Limit orders
- Stop orders
- Order tracking

### Portfolio Management

#### PortfolioAnalytics
Performance measurement:
- Return calculations
- Risk metrics
- Attribution analysis
- Benchmark comparison

#### PortfolioMonitor
Real-time monitoring:
- Alert generation
- Limit checking
- Performance tracking
- Risk monitoring

### Portfolio Optimization

#### Black-Litterman
Bayesian portfolio optimization:
- Market equilibrium
- View incorporation
- Uncertainty quantification
- Dynamic rebalancing

#### Hierarchical Risk Parity
Clustering-based allocation:
- Correlation clustering
- Risk contribution
- Diversification metrics
- Robust to estimation error

## Usage Examples

### Strategy Implementation
```python
from trading.strategies import MomentumStrategy, MeanReversionStrategy
from trading.risk_manager import RiskManager
from trading.adaptive_strategy_manager import AdaptiveStrategyManager

# Initialize components
risk_manager = RiskManager(initial_capital=100000)
momentum = MomentumStrategy(risk_manager)
mean_rev = MeanReversionStrategy(risk_manager)

# Strategy manager
manager = AdaptiveStrategyManager(risk_manager)
manager.add_strategy('momentum', momentum)
manager.add_strategy('mean_reversion', mean_rev)

# Generate signals
market_data = get_market_data()
signal = manager.get_best_signal(market_data)
```

### Risk Management
```python
from trading.risk_manager import RiskManager
from trading.position_sizer import PositionSizer

# Setup risk management
risk_manager = RiskManager(
    initial_capital=100000,
    max_position_size=0.1,
    max_drawdown=0.2
)

# Calculate position size
sizer = PositionSizer(risk_manager)
position_size = sizer.calculate_position_size(
    signal=signal,
    market_data=market_data,
    method='volatility_adjusted'
)

# Check risk limits
if risk_manager.check_risk_limits(symbol, position_size):
    execute_trade(symbol, position_size)
```

### Portfolio Optimization
```python
from trading.optimization import BlackLittermanOptimizer, HierarchicalRiskParity

# Black-Litterman optimization
bl_optimizer = BlackLittermanOptimizer()
views = generate_market_views()
optimal_weights = bl_optimizer.optimize(
    returns_data=returns,
    views=views,
    market_caps=market_caps
)

# HRP optimization
hrp = HierarchicalRiskParity()
hrp_weights = hrp.calculate_weights(returns_data)
```

## Configuration

### Strategy Parameters
```yaml
strategies:
  momentum:
    enabled: true
    lookback_period: 20
    entry_threshold: 2.0
    exit_threshold: 0.5
    
  mean_reversion:
    enabled: true
    lookback_period: 30
    entry_z_score: 2.5
    exit_z_score: 0.5
```

### Risk Parameters
```yaml
risk:
  max_position_size: 0.1
  max_drawdown: 0.2
  stop_loss: 0.02
  take_profit: 0.05
  max_correlation: 0.7
```

### Execution Parameters
```yaml
execution:
  slippage_model: 'linear'
  max_slippage: 0.001
  order_timeout: 30
  retry_attempts: 3
```

## Advanced Features

### Market Regime Detection
- Hidden Markov Models
- Clustering algorithms
- Volatility regimes
- Trend detection

### Dynamic Hedging
- Portfolio protection
- Options strategies
- Correlation hedging
- Tail risk management

### Adaptive Strategy Allocation
- Performance-based weighting
- Regime-based allocation
- Risk budgeting
- Machine learning integration

## Performance Metrics

### Strategy Metrics
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Profit factor

### Portfolio Metrics
- Total return
- Volatility
- Beta
- Alpha
- Information ratio

### Risk Metrics
- Value at Risk (VaR)
- Conditional VaR
- Maximum drawdown
- Correlation matrix
- Concentration risk

## Backtesting

The module supports comprehensive backtesting:
```python
from trading.backtesting import Backtester

backtester = Backtester(
    strategies=[momentum, mean_reversion],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

results = backtester.run(
    data=historical_data,
    initial_capital=100000
)

backtester.generate_report(results)
```

## Live Trading

Seamless transition from backtest to live:
```python
# Same strategy works for both
strategy = MomentumStrategy(risk_manager)

# Backtest mode
backtest_signal = strategy.generate_signal(historical_data)

# Live mode
live_signal = strategy.generate_signal(live_data)
```

## Testing

Comprehensive test suite:
```bash
# Run all trading tests
pytest tests/trading/

# Specific components
pytest tests/trading/strategies/
pytest tests/trading/risk_manager/
pytest tests/trading/execution/
```

## Best Practices

1. **Risk First**: Always check risk limits before execution
2. **Modular Design**: Strategies should be independent
3. **Error Handling**: Graceful degradation on failures
4. **Logging**: Comprehensive trade and decision logging
5. **Testing**: Both unit tests and integration tests

## Contributing

When adding to the trading module:
1. Follow existing patterns
2. Add comprehensive tests
3. Document parameters
4. Include backtesting support
5. Consider risk implications
