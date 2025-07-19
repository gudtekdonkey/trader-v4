"""
Trading Dashboard - Flask web application for monitoring
Provides real-time trading performance visualization, position tracking,
and risk metrics through a web interface.

File: app.py
Modified: 2025-07-15
"""

from flask import Flask, render_template, jsonify
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager
from src.utils.config import Config

app = Flask(__name__)
config = Config()
db = DatabaseManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    summary = db.get_performance_summary(days=30)
    
    return jsonify({
        'trade_stats': summary['trade_statistics'],
        'metrics': summary['performance_metrics'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/positions')
def get_positions():
    """Get current positions"""
    positions = db.get_open_positions()
    
    return jsonify({
        'positions': positions,
        'count': len(positions),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/trades/<int:days>')
def get_trades(days):
    """Get trade history"""
    trades = db.get_trade_history(days=days)
    
    if not trades.empty:
        trades['timestamp'] = trades['timestamp'].astype(str)
        trades_dict = trades.to_dict('records')
    else:
        trades_dict = []
    
    return jsonify({
        'trades': trades_dict,
        'count': len(trades_dict),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/equity_curve')
def get_equity_curve():
    """Get equity curve data"""
    # Get trades with cumulative PnL
    trades = db.get_trade_history(days=90)
    
    if not trades.empty and 'pnl' in trades.columns:
        trades = trades.sort_values('timestamp')
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        
        # Create equity curve
        initial_capital = config.get('trading.initial_capital', 100000)
        trades['equity'] = initial_capital + trades['cumulative_pnl']
        
        equity_data = [
            {
                'timestamp': row['timestamp'].isoformat(),
                'equity': row['equity'],
                'pnl': row['pnl']
            }
            for _, row in trades.iterrows()
        ]
    else:
        equity_data = []
    
    return jsonify({
        'data': equity_data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/risk_metrics')
def get_risk_metrics():
    """Get current risk metrics"""
    # This would connect to the live risk manager
    # For now, return sample data
    return jsonify({
        'total_exposure': 50000,
        'position_count': 3,
        'current_drawdown': 0.05,
        'max_drawdown': 0.08,
        'risk_score': 45,
        'var_95': 2500,
        'sharpe_ratio': 1.8,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.get('monitoring.dashboard_port', 8000), debug=True)
