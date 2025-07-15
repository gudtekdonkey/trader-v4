"""
LSTM with Multi-Head Self-Attention for Cryptocurrency Price Prediction

This module implements an advanced LSTM architecture enhanced with:
- Multi-head self-attention mechanism
- Bidirectional LSTM layers
- Dropout uncertainty estimation
- Global pooling strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention layer for sequence modeling.
    
    This layer implements scaled dot-product attention with multiple heads,
    allowing the model to attend to different aspects of the input sequence
    simultaneously.
    
    Attributes:
        hidden_dim: Dimension of hidden states
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        """
        Initialize the attention layer.
        
        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, \
            "Hidden dimension must be divisible by number of heads"
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the attention layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as input
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations and reshape for multi-head attention
        Q = self.query(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        K = self.key(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        V = self.value(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # Final linear transformation
        output = self.fc_out(context)
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        return self.layer_norm(output + x)


class AttentionLSTM(nn.Module):
    """
    LSTM with multi-head self-attention for cryptocurrency price prediction.
    
    This model combines bidirectional LSTM layers with self-attention
    mechanisms to capture both local and global temporal dependencies
    in cryptocurrency price data.
    
    Architecture:
        1. Input projection layer
        2. Bidirectional LSTM layers
        3. Multiple self-attention layers
        4. Global pooling (max + average)
        5. Feed-forward output layers
    
    Attributes:
        hidden_dim: Hidden dimension size
        num_layers: Number of LSTM layers
        input_projection: Linear layer for input transformation
        lstm: Bidirectional LSTM module
        attention_layers: List of attention layers
        output layers: Feed-forward layers for final prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 1,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        """
        Initialize the AttentionLSTM model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            output_dim: Output dimension (default 1 for price prediction)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(AttentionLSTM, self).__init__()
        
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
        
        # Attention layers (input dim is 2*hidden_dim due to bidirectional)
        self.attention_layers = nn.ModuleList([
            AttentionLayer(hidden_dim * 2, num_heads) 
            for _ in range(2)
        ])
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)  # *4 for concat pooling
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (predictions, attention_weights)
            attention_weights is None if return_attention is False
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        x = self.leaky_relu(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention layers
        attended = lstm_out
        attention_weights = []
        
        for attention_layer in self.attention_layers:
            attended = attention_layer(attended)
            if return_attention:
                attention_weights.append(attended)
        
        # Global max pooling and average pooling
        max_pooled = torch.max(attended, dim=1)[0]
        avg_pooled = torch.mean(attended, dim=1)
        
        # Concatenate pooled features
        pooled = torch.cat([max_pooled, avg_pooled], dim=-1)
        
        # Final predictions through feed-forward layers
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        predictions = self.fc3(out)
        
        if return_attention:
            return predictions, attention_weights
        return predictions, None
    
    def predict_with_confidence(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions with confidence intervals using dropout uncertainty.
        
        This method uses Monte Carlo dropout to estimate prediction uncertainty
        by running multiple forward passes with dropout enabled.
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        self.train()  # Enable dropout for uncertainty estimation
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred, _ = self.forward(x)
                predictions.append(pred)
        
        # Stack predictions and calculate statistics
        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        
        self.eval()  # Return to evaluation mode
        return mean_prediction, std_prediction