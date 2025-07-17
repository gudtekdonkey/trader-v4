import time
import requests
import numpy as np
import json
import os
from datetime import datetime

def get_currency_historical_data(api_key: str, coin_address: str, interval: str):

    current_unix_time = int(time.time())

    interval_time_calc = {
        "1m": int(time.time() - 1000*60),
        "3m": int(time.time() - 1000*3*60),
        "5m": int(time.time() - 1000*5*60),
        "15m": int(time.time() - 1000*15*60),
        "30m": int(time.time() - 1000*30*60),
        "1H": int(time.time() - 1000*1*60*60),
        "2H": int(time.time() - 1000*2*60*60),
        "4H": int(time.time() - 1000*4*60*60),
        "6H": int(time.time()  - 1000*6*60*60),
        "8H": int(time.time() - 1000*8*60*60),
        "12H": int(time.time() - 1000*12*60*60),
        "1D": int(time.time() - 1000*1*24*60*60),
        "3D": int(time.time() - 1000*3*24*60*60),
        "1W": int(time.time() - 1000*1*7*24*60*60),
        "1M": 0
    }

    query = f"https://public-api.birdeye.so/defi/ohlcv?address={coin_address}&type={interval}&currency=usd&time_from={interval_time_calc[interval]}&time_to={current_unix_time}"
    
    
    headers = {
        "accept": "application/json",
        "x-chain": "solana",
        "X-API-Key": api_key
    }

    response = requests.get(query, headers=headers)
    
    response = response.json()
    print(response)
    candles = response["data"]["items"]

    open_numpy = np.array([], dtype=float)
    high_numpy = np.array([], dtype=float)
    low_numpy = np.array([], dtype=float)
    close_numpy = np.array([], dtype=float)
    volume_numpy = np.array([], dtype=float)

    for candle in candles:
        open_numpy = np.append(open_numpy, candle["o"])
        high_numpy = np.append(high_numpy, candle["h"])
        low_numpy = np.append(low_numpy, candle["l"])
        close_numpy = np.append(close_numpy, candle["c"])
        volume_numpy = np.append(volume_numpy, candle["v"])
    
    # Save the data to file
    save_historical_data(
        coin_address=coin_address,
        interval=interval,
        candles=candles,
        open_data=open_numpy,
        high_data=high_numpy,
        low_data=low_numpy,
        close_data=close_numpy,
        volume_data=volume_numpy,
        raw_response=response
    )
    
    return open_numpy, high_numpy, low_numpy, close_numpy, volume_numpy


def save_historical_data(coin_address, interval, candles, open_data, high_data, low_data, close_data, volume_data, raw_response):
    """
    Save historical data in a format easily retrievable in Python
    """
    # Create directory if it doesn't exist
    save_dir = "../data/historical"
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data structure
    data_to_save = {
        "metadata": {
            "coin_address": coin_address,
            "interval": interval,
            "timestamp": datetime.now().isoformat(),
            "unix_timestamp": int(time.time()),
            "data_points": len(candles)
        },
        "arrays": {
            "open": open_data.tolist(),
            "high": high_data.tolist(),
            "low": low_data.tolist(),
            "close": close_data.tolist(),
            "volume": volume_data.tolist()
        },
        "candles": candles,
        "raw_response": raw_response
    }
    
    # Create filename with timestamp and coin info
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"birdeye_historical_data_{coin_address[:8]}_{interval}_{timestamp_str}.json"
    filepath = os.path.join(save_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Data saved to: {filepath}")
    
    # Also save a "latest" file for easy access
    latest_filepath = os.path.join(save_dir, "birdeye_historical_data_latest.json")
    with open(latest_filepath, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Latest data also saved to: {latest_filepath}")


def load_historical_data(filepath=None):
    """
    Helper function to load saved historical data in Python
    
    Usage:
        # Load latest data
        data = load_historical_data()
        
        # Load specific file
        data = load_historical_data("../data/historical/birdeye_historical_data_12345678_1H_20240115_120000.json")
        
        # Access arrays as numpy arrays
        open_prices = np.array(data['arrays']['open'])
        close_prices = np.array(data['arrays']['close'])
    """
    if filepath is None:
        filepath = "../data/historical/birdeye_historical_data_latest.json"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert arrays back to numpy if needed
    for key in ['open', 'high', 'low', 'close', 'volume']:
        if key in data['arrays']:
            data['arrays'][key + '_numpy'] = np.array(data['arrays'][key])
    
    return data