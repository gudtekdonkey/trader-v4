import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Set up logger with console and file handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - rotating by size
    file_handler = RotatingFileHandler(
        log_dir / 'trading_bot.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Trade log handler - daily rotation
    trade_handler = TimedRotatingFileHandler(
        log_dir / 'trades.log',
        when='midnight',
        interval=1,
        backupCount=30
    )
    trade_handler.setLevel(logging.INFO)
    trade_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    trade_handler.setFormatter(trade_formatter)
    
    # Create trade logger
    trade_logger = logging.getLogger('trades')
    trade_logger.setLevel(logging.INFO)
    trade_logger.addHandler(trade_handler)
    
    return logger

# Convenience functions for trade logging
def log_trade(symbol: str, side: str, size: float, price: float, 
              pnl: float = None, reason: str = None):
    """Log trade execution"""
    trade_logger = logging.getLogger('trades')
    
    message = f"TRADE - Symbol: {symbol}, Side: {side}, Size: {size}, Price: {price}"
    
    if pnl is not None:
        message += f", PnL: {pnl:.2f}"
    
    if reason:
        message += f", Reason: {reason}"
    
    trade_logger.info(message)

def log_position_update(symbol: str, action: str, details: Dict):
    """Log position updates"""
    trade_logger = logging.getLogger('trades')
    
    message = f"POSITION - Symbol: {symbol}, Action: {action}"
    
    for key, value in details.items():
        message += f", {key}: {value}"
    
    trade_logger.info(message)
