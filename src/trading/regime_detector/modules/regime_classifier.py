"""
Regime Classifier Module

Neural network classifier for market regime prediction and classification.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class RegimeClassifier(nn.Module):
    """
    Neural network classifier for regime prediction.
    
    This module implements a multi-layer neural network for classifying
    market regimes based on extracted features.
    """
    
    def __init__(self, n_regimes: int, input_dim: int = 20):
        """
        Initialize the regime classifier.
        
        Args:
            n_regimes: Number of regime classes
            input_dim: Input feature dimension
        """
        super().__init__()
        
        self.n_regimes = n_regimes
        self.input_dim = input_dim
        
        try:
            # Build network architecture
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(32),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, n_regimes),
                nn.Softmax(dim=-1)
            )
            
            # Initialize weights
            self._init_weights()
            
            logger.info(f"RegimeClassifier initialized for {n_regimes} regimes")
            
        except Exception as e:
            logger.error(f"Failed to build classifier: {e}")
            # Return simple classifier as fallback
            self.network = nn.Sequential(
                nn.Linear(input_dim, n_regimes),
                nn.Softmax(dim=-1)
            )
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input features tensor
            
        Returns:
            Regime probabilities
        """
        try:
            # Validate input
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("NaN/Inf in classifier input")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Forward pass
            output = self.network(x)
            
            # Validate output
            if torch.isnan(output).any():
                logger.warning("NaN in classifier output")
                # Return uniform probabilities
                batch_size = x.size(0)
                return torch.ones(batch_size, self.n_regimes) / self.n_regimes
            
            return output
            
        except Exception as e:
            logger.error(f"Error in classifier forward pass: {e}")
            # Return uniform probabilities
            batch_size = x.size(0)
            return torch.ones(batch_size, self.n_regimes) / self.n_regimes
    
    def predict(self, features: torch.Tensor) -> int:
        """
        Predict the most likely regime.
        
        Args:
            features: Input features
            
        Returns:
            Predicted regime index
        """
        with torch.no_grad():
            probs = self.forward(features)
            return torch.argmax(probs, dim=-1).item()
    
    def get_confidence(self, features: torch.Tensor) -> float:
        """
        Get confidence of the prediction.
        
        Args:
            features: Input features
            
        Returns:
            Confidence score (max probability)
        """
        with torch.no_grad():
            probs = self.forward(features)
            return torch.max(probs).item()
