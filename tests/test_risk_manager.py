import unittest
from src.trading.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    """Test risk management system"""
    
    def setUp(self):
        """Set up test environment"""
        self.risk_manager = RiskManager(initial_capital=100000)
        
    def test_initialization(self):
        """Test risk manager initialization"""
        self.assertEqual(self.risk_manager.initial_capital, 100000)
        self.assertEqual(self.risk_manager.current_capital, 100000)
        self.assertEqual(len(self.risk_manager.positions), 0)
        
    def test_position_sizing(self):
        """Test position size calculation"""
        # Test basic position sizing
        size = self.risk_manager.calculate_position_size(
            entry_price=50000,
            stop_loss=49000,
            symbol='BTC-USD'
        )
        
        # With 2% risk and $1000 price risk, size should be reasonable
        self.assertGreater(size, 0)
        self.assertLess(size, 10)  # Should be less than 10 BTC
        
    def test_risk_checks(self):
        """Test pre-trade risk checks"""
        # Test valid trade
        can_trade, reason = self.risk_manager.check_pre_trade_risk(
            symbol='BTC-USD',
            side='buy',
            size=1,
            price=50000,
            stop_loss=49000
        )
        
        self.assertTrue(can_trade)
        self.assertEqual(reason, "Risk check passed")
        
        # Test position limit
        for i in range(10):
            self.risk_manager.add_position(
                symbol=f'TEST{i}',
                side='buy',
                size=0.1,
                entry_price=50000
            )
        
        # Should fail on 11th position
        can_trade, reason = self.risk_manager.check_pre_trade_risk(
            symbol='TEST11',
            side='buy',
            size=0.1,
            price=50000
        )
        
        self.assertFalse(can_trade)
        self.assertEqual(reason, "Position limit reached")
        
    def test_pnl_calculation(self):
        """Test PnL calculations"""
        # Add a position
        self.risk_manager.add_position(
            symbol='BTC-USD',
            side='long',
            size=1,
            entry_price=50000
        )
        
        # Update price (10% gain)
        self.risk_manager.update_position_price('BTC-USD', 55000)
        
        position = self.risk_manager.positions['BTC-USD']
        self.assertEqual(position['unrealized_pnl'], 5000)
        
        # Close position
        self.risk_manager.close_position('BTC-USD', 55000)
        
        # Check capital updated
        self.assertEqual(self.risk_manager.current_capital, 105000)
        
    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        # Add some trades to history
        for i in range(10):
            self.risk_manager._update_daily_pnl(1000 if i % 2 == 0 else -500)
        
        # Calculate metrics
        metrics = self.risk_manager.calculate_risk_metrics()
        
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertIsNotNone(metrics.var_95)
        self.assertIsNotNone(metrics.risk_score)
        
        # Risk score should be between 0 and 100
        self.assertGreaterEqual(metrics.risk_score, 0)
        self.assertLessEqual(metrics.risk_score, 100)

if __name__ == '__main__':
    unittest.main()
