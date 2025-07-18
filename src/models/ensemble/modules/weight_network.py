"""
Adaptive weight network module for ensemble predictor.
Implements neural network for dynamic weight allocation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AdaptiveWeightNetwork(nn.Module):
    """
    Neural network for dynamic ensemble weight allocation with error handling.
    
    This network learns to assign weights to different models based on
    market conditions and features, enabling adaptive ensemble behavior.
    
    Attributes:
        num_models: Number of models in the ensemble
        feature_dim: Dimension of input features
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(self, num_models: int, feature_dim: int, hidden_dim: int = 64):
        """
        Initialize the adaptive weight network with validation.
        
        Args:
            num_models: Number of models to weight
            feature_dim: Dimension of market features
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        # Validate inputs
        if num_models <= 0:
            raise ValueError(f"Number of models must be positive, got {num_models}")
        if feature_dim <= 0:
            raise ValueError(f"Feature dimension must be positive, got {feature_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {hidden_dim}")
        
        try:
            self.fc1 = nn.Linear(feature_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_models)
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(0.2)
            
            # Initialize weights for stability
            self._init_weights()
            
            # Track network statistics
            self.call_count = 0
            self.nan_count = 0
            
            logger.info(f"AdaptiveWeightNetwork initialized for {num_models} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize AdaptiveWeightNetwork: {e}")
            raise
    
    def _init_weights(self):
        """Initialize weights for stability"""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate model weights with error handling.
        
        Args:
            x: Market features tensor
            
        Returns:
            Softmax weights for each model
        """
        self.call_count += 1
        
        # Validate input
        if x is None:
            raise ValueError("Input tensor cannot be None")
        
        # Handle NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN/Inf in weight network input")
            self.nan_count += 1
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        try:
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            weights = self.softmax(self.fc3(x))
            
            # Validate output
            if torch.isnan(weights).any():
                logger.warning("NaN in adaptive weights, using uniform weights")
                batch_size = x.size(0)
                num_models = self.fc3.out_features
                weights = torch.ones(batch_size, num_models) / num_models
                weights = weights.to(x.device)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in weight generation: {e}")
            # Return uniform weights
            batch_size = x.size(0)
            num_models = self.fc3.out_features
            return torch.ones(batch_size, num_models) / num_models
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network usage statistics"""
        return {
            'call_count': self.call_count,
            'nan_count': self.nan_count,
            'nan_rate': self.nan_count / self.call_count if self.call_count > 0 else 0
        }
    
    def reset_parameters(self):
        """Reset network parameters"""
        self._init_weights()
        self.call_count = 0
        self.nan_count = 0
        logger.info("AdaptiveWeightNetwork parameters reset")
