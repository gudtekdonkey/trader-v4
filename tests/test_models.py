import unittest
import torch
import numpy as np
import pandas as pd
from src.models.lstm_attention import AttentionLSTM
from models.ensemble.ensemble import EnsemblePredictor
from src.models.regime_detector import MarketRegimeDetector

class TestModels(unittest.TestCase):
    """Test ML models"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'open': np.random.randn(100) * 100 + 50000,
            'high': np.random.randn(100) * 100 + 50100,
            'low': np.random.randn(100) * 100 + 49900,
            'close': np.random.randn(100) * 100 + 50000,
            'volume': np.random.rand(100) * 1000000
        })
        self.sample_data.index = pd.date_range('2024-01-01', periods=100, freq='1H')
        
    def test_attention_lstm(self):
        """Test AttentionLSTM model"""
        model = AttentionLSTM(input_dim=10, hidden_dim=64, num_layers=2)
        
        # Test forward pass
        batch_size = 8
        seq_len = 60
        input_dim = 10
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output, attention = model(x)
        
        self.assertEqual(output.shape, (batch_size, 1))
        self.assertIsNone(attention)  # When return_attention=False
        
        # Test with attention
        output, attention = model(x, return_attention=True)
        self.assertIsNotNone(attention)
        
    def test_ensemble_predictor(self):
        """Test ensemble predictor"""
        ensemble = EnsemblePredictor(device='cpu')
        
        # Test initialization
        self.assertEqual(len(ensemble.lgb_models), 0)
        self.assertEqual(len(ensemble.xgb_models), 0)
        
    def test_regime_detector(self):
        """Test market regime detector"""
        detector = MarketRegimeDetector(n_regimes=3)
        
        # Test feature extraction
        features = detector.extract_regime_features(self.sample_data)
        self.assertEqual(features.shape[0], len(self.sample_data))
        
        # Test fitting
        detector.fit(self.sample_data)
        
        # Test regime detection
        regime_info = detector.detect_regime(self.sample_data)
        
        self.assertIn('regime', regime_info)
        self.assertIn('confidence', regime_info)
        self.assertIn('trading_mode', regime_info)

if __name__ == '__main__':
    unittest.main()
