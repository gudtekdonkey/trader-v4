import unittest
import pandas as pd
import numpy as np
from src.trading.strategies.momentum import MomentumStrategy
from src.trading.strategies.mean_reversion import MeanReversionStrategy
from trading.risk_manager.risk_manager import RiskManager

class TestStrategies(unittest.TestCase):
    """Test trading strategies"""
    
    def setUp(self):
        """Set up test environment"""
        self.risk_manager = RiskManager(initial_capital=100000)
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=200, freq='1H')
        self.sample_data = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50100,
            'low': np.random.randn(200).cumsum() + 49900,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.rand(200) * 1000000
        }, index=dates)
        
        # Ensure high > low
        self.sample_data['high'] = self.sample_data[['high', 'open', 'close']].max(axis=1)
        self.sample_data['low'] = self.sample_data[['low', 'open', 'close']].min(axis=1)
        
    def test_momentum_strategy(self):
        """Test momentum strategy"""
        strategy = MomentumStrategy(self.risk_manager)
        
        # Test signal generation
        signals = strategy.analyze(self.sample_data)
        
        # Signals should be a list
        self.assertIsInstance(signals, list)
        
        # If signals exist, check structure
        if signals:
            signal = signals[0]
            self.assertIn('direction', signal.__dict__)
            self.assertIn('entry_price', signal.__dict__)
            self.assertIn('stop_loss', signal.__dict__)
            self.assertIn('confidence', signal.__dict__)
    
    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy"""
        strategy = MeanReversionStrategy(self.risk_manager)
        
        # Add some mean-reverting behavior
        self.sample_data['close'] = 50000 + 1000 * np.sin(np.arange(200) * 0.1)
        
        # Test signal generation
        signals = strategy.analyze(self.sample_data)
        
        # Signals should be a list
        self.assertIsInstance(signals, list)
        
        # Strategy metrics should be initialized
        self.assertIn('total_trades', strategy.stats)
        self.assertIn('win_rate', strategy.stats)

if __name__ == '__main__':
    unittest.main()
