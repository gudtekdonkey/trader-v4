# Data Module

## Overview
The data module handles all data collection, preprocessing, and feature engineering for the trading system. It provides real-time market data access and historical data management.

## Structure
```
data/
├── __init__.py              # Package initialization
├── preprocessor.py          # Data preprocessing utilities
├── README.md               # This file
├── collector/              # Data collection submodule
│   ├── __init__.py
│   ├── collector.py        # Main data collector
│   └── modules/
│       ├── connection_manager.py  # WebSocket management
│       ├── data_processor.py      # Message processing
│       ├── redis_manager.py       # Redis operations
│       └── ...
└── feature_engineer/       # Feature engineering submodule
    ├── __init__.py
    ├── feature_engineer.py  # Main feature engineering
    └── modules/
        ├── time_features.py       # Time-based features
        ├── technical_features.py   # Technical indicators
        └── ...
```

## Components

### DataCollector
Real-time data collection via WebSocket:
- Order book snapshots
- Trade executions
- Funding rates
- Market statistics

### DataPreprocessor
Data cleaning and preparation:
- Missing value handling
- Outlier detection
- Normalization
- Time alignment

### FeatureEngineer
Advanced feature creation:
- Technical indicators
- Market microstructure
- Sentiment analysis
- Cross-asset features

## Key Features

### 1. Real-time Data Collection
```python
from data import DataCollector

collector = DataCollector(redis_host='localhost')
await collector.subscribe_symbol('BTC-USD')

# Get latest data
orderbook = await collector.get_orderbook('BTC-USD')
trades = await collector.get_recent_trades('BTC-USD')
```

### 2. Data Preprocessing
```python
from data import DataPreprocessor

preprocessor = DataPreprocessor()
clean_data = preprocessor.prepare_ohlcv_data(raw_data)
normalized = preprocessor.normalize_features(clean_data)
```

### 3. Feature Engineering
```python
from data import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.engineer_all_features(market_data)
```

## Data Sources

### Market Data
- **Order Books**: L2 depth with configurable levels
- **Trades**: Real-time execution data
- **OHLCV**: Candlestick data at multiple timeframes
- **Funding**: Perpetual funding rates

### Alternative Data
- On-chain metrics
- Social sentiment
- News sentiment
- Correlation data

## Redis Schema

Data stored in Redis with TTL:
```
# Order book
orderbook:{symbol} -> JSON snapshot

# Trades
trades:{symbol} -> List of recent trades

# OHLCV
ohlcv:{symbol}:{timeframe} -> Candlestick data

# Features
features:{symbol} -> Computed features
```

## Feature Categories

### Technical Features
- Moving averages (SMA, EMA, WMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility measures (ATR, Bollinger Bands)
- Volume indicators (OBV, VWAP)

### Microstructure Features
- Bid-ask spread
- Order flow imbalance
- Trade size distribution
- Market depth metrics

### Statistical Features
- Rolling statistics
- Autocorrelation
- Entropy measures
- Distribution moments

### Time Features
- Hour of day
- Day of week
- Month effects
- Holiday indicators

## Performance Optimization

### Caching Strategy
- Redis for real-time data
- Local cache for computed features
- TTL-based expiration
- Memory-mapped files for large datasets

### Parallel Processing
- Async I/O for data collection
- Multi-threading for feature computation
- Batch processing for efficiency

## Data Quality

### Validation Checks
- Timestamp consistency
- Price sanity checks
- Volume validation
- Spread reasonableness

### Error Handling
- Automatic reconnection
- Data gap detection
- Fallback mechanisms
- Alert generation

## Usage Examples

### Real-time Trading
```python
# Initialize components
collector = DataCollector()
preprocessor = DataPreprocessor()
engineer = FeatureEngineer()

# Collect and process data
raw_data = await collector.get_market_data('BTC-USD')
clean_data = preprocessor.prepare_data(raw_data)
features = engineer.engineer_features(clean_data)
```

### Backtesting
```python
# Load historical data
historical = await collector.get_historical_data(
    symbol='BTC-USD',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Process for backtesting
processed = preprocessor.prepare_backtest_data(historical)
```

## Configuration

Key settings in config.yaml:
```yaml
data:
  redis_host: localhost
  redis_port: 6379
  max_reconnect_attempts: 5
  reconnect_delay: 5
  cache_ttl: 300
  orderbook_depth: 20
  trade_history_size: 1000
```

## Monitoring

Data health metrics:
- Connection status
- Data latency
- Message rates
- Cache hit rates
- Error counts

## Testing

Run data module tests:
```bash
pytest tests/data/
```

## Contributing

When extending the data module:
1. Maintain data integrity
2. Handle edge cases
3. Document data formats
4. Add validation tests
5. Consider performance impact
