"""Attention layer implementation for LSTM models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention layer for sequence modeling with error handling.
    
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
        Initialize the attention layer with validation.
        
        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super(AttentionLayer, self).__init__()
        
        # [ERROR-HANDLING] Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"Number of heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden dimension {hidden_dim} must be divisible by number of heads {num_heads}")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # [ERROR-HANDLING] Validate head dimension
        if self.head_dim < 4:
            logger.warning(f"Very small head dimension {self.head_dim}, may lead to poor performance")
        
        # Linear transformations for Q, K, V
        try:
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, hidden_dim)
            
            self.dropout = nn.Dropout(0.1)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
            # [ERROR-HANDLING] Initialize weights to prevent gradient issues
            self._init_weights()
            
        except Exception as e:
            logger.error(f"Failed to initialize attention layers: {e}")
            raise
        
        logger.info(f"AttentionLayer initialized with hidden_dim={hidden_dim}, num_heads={num_heads}")
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization for stability"""
        for module in [self.query, self.key, self.value, self.fc_out]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the attention layer with comprehensive error handling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as input
        """
        # [ERROR-HANDLING] Validate input
        if x is None:
            raise ValueError("Input tensor cannot be None")
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"Input hidden dimension {x.size(-1)} doesn't match expected {self.hidden_dim}")
        
        # [ERROR-HANDLING] Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or Inf detected in attention input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        batch_size, seq_len, _ = x.size()
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error in linear transformations: {e}")
            # [ERROR-HANDLING] Return input unchanged as fallback
            return x
        
        try:
            # Scaled dot-product attention
            # [ERROR-HANDLING] Use more stable computation
            scale = np.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            
            # [ERROR-HANDLING] Prevent overflow in softmax
            scores = torch.clamp(scores, min=-1e4, max=1e4)
            
            # Apply mask if provided
            if mask is not None:
                # [ERROR-HANDLING] Validate mask shape
                if mask.shape[-1] != seq_len:
                    logger.warning(f"Mask shape {mask.shape} doesn't match sequence length {seq_len}")
                else:
                    scores = scores.masked_fill(mask == 0, -1e9)
            
            # Softmax and dropout
            attention_weights = F.softmax(scores, dim=-1)
            
            # [ERROR-HANDLING] Check for NaN in attention weights
            if torch.isnan(attention_weights).any():
                logger.warning("NaN in attention weights, using uniform attention")
                attention_weights = torch.ones_like(attention_weights) / seq_len
            
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
            # [ERROR-HANDLING] Ensure stable addition
            output = self.layer_norm(output + x)
            
            # [ERROR-HANDLING] Final NaN/Inf check
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.warning("NaN or Inf in attention output, returning input")
                return x
            
            return output
            
        except Exception as e:
            logger.error(f"Error in attention computation: {e}")
            # [ERROR-HANDLING] Return input unchanged as fallback
            return x
