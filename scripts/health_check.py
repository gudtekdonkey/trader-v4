#!/usr/bin/env python3
"""
Health check script for trading bot
Verifies all components are working correctly
"""

import os
import sys
import psycopg2
import redis
import requests
from datetime import datetime
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama
init()

# Load environment variables
load_dotenv()


def check_database():
    """Check PostgreSQL connection"""
    try:
        db_url = os.getenv('DATABASE_URL')
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def check_redis():
    """Check Redis connection"""
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        r = redis.from_url(redis_url)
        r.ping()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def check_hyperliquid():
    """Check Hyperliquid API connection"""
    try:
        api_url = os.getenv('HYPERLIQUID_API_URL', 'https://api.hyperliquid.xyz')
        response = requests.get(f"{api_url}/info", timeout=5)
        if response.status_code == 200:
            return True, "Connected"
        else:
            return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)


def check_telegram():
    """Check Telegram bot connection"""
    try:
        token = os.getenv('TELEGRAM_TOKEN')
        if not token:
            return False, "Token not configured"
        
        response = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=5)
        if response.status_code == 200:
            bot_info = response.json()
            return True, f"Bot: @{bot_info['result']['username']}"
        else:
            return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)


def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        stat = shutil.disk_usage("/")
        free_gb = stat.free / (1024 ** 3)
        total_gb = stat.total / (1024 ** 3)
        percent_used = (stat.total - stat.free) / stat.total * 100
        
        if free_gb < 1:
            return False, f"Low disk space: {free_gb:.1f}GB free"
        else:
            return True, f"{free_gb:.1f}GB free ({percent_used:.1f}% used)"
    except Exception as e:
        return False, str(e)


def check_python_packages():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'sqlalchemy', 'redis', 'ccxt',
        'ta', 'plotly', 'flask', 'prometheus_client'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    else:
        return True, "All packages installed"


def print_result(name, success, message):
    """Print formatted result"""
    if success:
        status = f"{Fore.GREEN}[✓]{Style.RESET_ALL}"
    else:
        status = f"{Fore.RED}[✗]{Style.RESET_ALL}"
    
    print(f"{status} {name:<20} {message}")


def main():
    """Run all health checks"""
    print("================================================")
    print("Trading Bot Health Check")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("================================================\n")
    
    checks = [
        ("PostgreSQL", check_database),
        ("Redis", check_redis),
        ("Hyperliquid API", check_hyperliquid),
        ("Telegram Bot", check_telegram),
        ("Disk Space", check_disk_space),
        ("Python Packages", check_python_packages),
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        success, message = check_func()
        print_result(name, success, message)
        if not success:
            all_passed = False
    
    print("\n================================================")
    
    if all_passed:
        print(f"{Fore.GREEN}All checks passed!{Style.RESET_ALL}")
        sys.exit(0)
    else:
        print(f"{Fore.RED}Some checks failed!{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()