-- Initialize trading database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create tables
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    fee DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    strategy VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    exchange VARCHAR(50) DEFAULT 'hyperliquid',
    order_id VARCHAR(100),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    side VARCHAR(10) NOT NULL,
    leverage DECIMAL(10, 2) DEFAULT 1,
    margin DECIMAL(20, 8),
    liquidation_price DECIMAL(20, 8),
    strategy VARCHAR(100),
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'open',
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    UNIQUE(symbol, timestamp, timeframe)
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    total_pnl DECIMAL(20, 8) NOT NULL,
    win_rate DECIMAL(5, 2),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win DECIMAL(20, 8),
    avg_loss DECIMAL(20, 8),
    largest_win DECIMAL(20, 8),
    largest_loss DECIMAL(20, 8),
    profit_factor DECIMAL(10, 4),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    var_95 DECIMAL(20, 8),
    cvar_95 DECIMAL(20, 8),
    portfolio_beta DECIMAL(10, 4),
    correlation_matrix JSONB,
    position_sizes JSONB,
    risk_score DECIMAL(5, 2),
    metadata JSONB
);

-- Create indexes
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp);
CREATE INDEX idx_trades_strategy ON trades(strategy);
CREATE INDEX idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX idx_market_data_symbol_timeframe ON market_data(symbol, timeframe, timestamp DESC);
CREATE INDEX idx_performance_timestamp ON performance_metrics(timestamp DESC);

-- Create views
CREATE VIEW current_positions AS
SELECT 
    symbol,
    side,
    quantity,
    entry_price,
    current_price,
    unrealized_pnl,
    leverage,
    strategy,
    opened_at
FROM positions
WHERE status = 'open';

CREATE VIEW daily_performance AS
SELECT 
    DATE(timestamp) as date,
    SUM(realized_pnl) as daily_pnl,
    COUNT(*) as trades_count,
    AVG(realized_pnl) as avg_pnl_per_trade
FROM trades
GROUP BY DATE(timestamp)
ORDER BY date DESC;