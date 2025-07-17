#!/usr/bin/env python3
"""
Download sample market data for testing and development
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
from sqlalchemy import create_engine
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://trader:trader_password@localhost:5432/trading')
DATA_DIR = '../data'
PARENT_DIR = '/historical/sample'
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAMES = ['1h', '4h', '1d']
DAYS_TO_DOWNLOAD = 365


def download_historical_data(exchange, symbol, timeframe, days):
    """Download historical OHLCV data"""
    print(f"Downloading {symbol} {timeframe} data...")
    
    # Calculate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Convert to milliseconds
    since = int(start_time.timestamp() * 1000)
    
    # Download data
    all_candles = []
    
    while since < int(end_time.timestamp() * 1000):
        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )
            
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['symbol'] = symbol.replace('/', '-')
    df['timeframe'] = timeframe
    
    return df


def save_to_database(df, table_name='market_data'):
    """Save data to database"""
    engine = create_engine(DATABASE_URL)
    
    try:
        df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"Saved {len(df)} records to database")
    except Exception as e:
        print(f"Error saving to database: {e}")


def save_to_csv(df, symbol, timeframe):
    """Save data to CSV file"""
    filename = f"{DATA_DIR}{PARENT_DIR}/{symbol.replace('/', '-')}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")


def generate_sample_trades(symbols, days=30):
    """Generate sample trade data for testing"""
    print("\nGenerating sample trade data...")
    
    trades = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for _ in range(1000):  # Generate 1000 sample trades
        # Random trade parameters
        symbol = np.random.choice(symbols)
        timestamp = start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        side = np.random.choice(['buy', 'sell'])
        price = np.random.uniform(30000, 60000) if 'BTC' in symbol else np.random.uniform(2000, 4000)
        quantity = np.random.uniform(0.001, 0.1)
        fee = price * quantity * 0.001
        realized_pnl = np.random.normal(0, 100)
        strategy = np.random.choice(['momentum', 'mean_reversion', 'arbitrage', 'market_making'])
        
        trades.append({
            'symbol': symbol.replace('/', '-'),
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'quantity': quantity,
            'fee': fee,
            'realized_pnl': realized_pnl,
            'strategy': strategy
        })
    
    trades_df = pd.DataFrame(trades)
    
    # Save to database
    engine = create_engine(DATABASE_URL)
    trades_df.to_sql('trades', engine, if_exists='append', index=False)
    
    print(f"Generated {len(trades_df)} sample trades")


def main():
    """Main function"""
    print("Starting data download...")
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize exchange (using Binance for sample data)
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })
    
    # Download data for each symbol and timeframe
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            try:
                df = download_historical_data(exchange, symbol, timeframe, DAYS_TO_DOWNLOAD)
                
                if not df.empty:
                    # Save to database
                    save_to_database(df)
                    
                    # Save to CSV
                    save_to_csv(df, symbol, timeframe)
                
            except Exception as e:
                print(f"Error processing {symbol} {timeframe}: {e}")
    
    # Generate sample trades
    generate_sample_trades([s.replace('/', '-') for s in SYMBOLS])
    
    print("\nData download completed!")


if __name__ == "__main__":
    main()