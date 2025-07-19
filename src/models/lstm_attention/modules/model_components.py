"""LSTM model components and architecture"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for sequence processing"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.2):
        """
        Initialize LSTM encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers (bidirectional doubles the output dimension)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Activation
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        # Input projection
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0)
        
        # LSTM weights - use orthogonal initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of LSTM encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (output, (hidden, cell)) where output has shape 
            (batch_size, seq_len, hidden_dim * 2)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Input projection
        x = self.input_projection(x)
        x = self.leaky_relu(x)
        
        # Check for NaN after projection
        if torch.isnan(x).any():
            logger.warning("NaN after input projection, reinitializing projection layer")
            self.input_projection.reset_parameters()
            x = self.input_projection(x)
            x = self.leaky_relu(x)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x, (h0, c0))
        lstm_out = self.layer_norm(lstm_out)
        
        # Check LSTM output
        if torch.isnan(lstm_out).any():
            logger.warning("NaN in LSTM output, using skip connection")
            lstm_out = x.repeat(1, 1, 2)  # Simulate bidirectional output
        
        return lstm_out, (hidden, cell)


class OutputHead(nn.Module):
    """Output head for final predictions"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize output head.
        
        Args:
            input_dim: Input dimension (after pooling)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(OutputHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of output head.
        
        Args:
            x: Input tensor after pooling
            
        Returns:
            Predictions tensor
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        predictions = self.fc3(out)
        
        # Check final predictions
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            logger.warning("NaN/Inf in predictions, returning zeros")
            predictions = torch.zeros_like(predictions)
        
        # Clip predictions to reasonable range
        predictions = torch.clamp(predictions, min=-100, max=100)
        
        return predictions


class TemporalPooling(nn.Module):
    """Temporal pooling for sequence aggregation"""
    
    def __init__(self):
        """Initialize temporal pooling"""
        super(TemporalPooling, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply both max and average pooling, then concatenate.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Pooled tensor of shape (batch_size, features * 2)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Handle empty sequences
        if x.size(1) == 0:
            return torch.zeros((batch_size, x.size(-1) * 2), device=device)
        
        # Global max pooling
        max_pooled = torch.max(x, dim=1)[0]
        
        # Global average pooling
        avg_pooled = torch.mean(x, dim=1)
        
        # Concatenate pooled features
        pooled = torch.cat([max_pooled, avg_pooled], dim=-1)
        
        # Check pooled features
        if torch.isnan(pooled).any():
            logger.warning("NaN in pooled features, using zeros")
            pooled = torch.zeros_like(pooled)
        
        return pooled
