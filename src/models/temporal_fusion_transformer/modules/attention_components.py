"""
Attention Components Module

Implements various attention mechanisms for the TFT model,
including multi-head attention with gating for temporal dependencies.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class MultiHeadAttentionWithGating(nn.Module):
    """
    Multi-head attention with gating mechanism for TFT.
    
    This combines standard multi-head attention with a gating mechanism
    to control information flow and improve gradient propagation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention with gating.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear projections
        """
        super().__init__()
        
        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"Number of heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} must be divisible by num_heads {num_heads}")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        try:
            # Standard multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                batch_first=True
            )
            
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(embed_dim)
            
            # Initialize weights
            self._init_weights()
            
        except Exception as e:
            logger.error(f"Failed to initialize MultiHeadAttentionWithGating: {e}")
            raise
    
    def _init_weights(self):
        """Initialize weights for stability"""
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with error handling.
        
        Args:
            query: Query tensor
            key: Key tensor (defaults to query for self-attention)
            value: Value tensor (defaults to query for self-attention)
            key_padding_mask: Mask for padding tokens
            attn_mask: Attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Validate inputs
        if torch.isnan(query).any() or torch.isinf(query).any():
            logger.warning("NaN/Inf in attention query")
            query = torch.nan_to_num(query, nan=0.0, posinf=1e6, neginf=-1e6)
            key = torch.nan_to_num(key, nan=0.0, posinf=1e6, neginf=-1e6)
            value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        
        try:
            # Apply multi-head attention
            attn_output, attn_weights = self.attention(
                query, key, value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
            
            # Check for NaN in attention output
            if torch.isnan(attn_output).any():
                logger.warning("NaN in attention output, using residual connection")
                attn_output = query
            
            # Apply gating
            gate_values = self.gate(query)
            gated_output = gate_values * attn_output + (1 - gate_values) * query
            
            # Layer normalization
            output = self.layer_norm(gated_output)
            
            # Final validation
            if torch.isnan(output).any():
                logger.warning("NaN in final attention output, using input")
                output = query
            
            return output, attn_weights
            
        except Exception as e:
            logger.error(f"Error in attention forward pass: {e}")
            # Return input as fallback
            return query, None


class TemporalSelfAttention(nn.Module):
    """
    Temporal self-attention specifically designed for time series.
    
    This module applies self-attention over the temporal dimension
    with positional encodings and temporal masks.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_length: int = 1000,
        dropout: float = 0.1
    ):
        """
        Initialize temporal self-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = MultiHeadAttentionWithGating(
            embed_dim, num_heads, dropout
        )
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            max_seq_length, embed_dim
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(
        self,
        max_seq_length: int,
        embed_dim: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            -(torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        try:
            # Add positional encoding
            seq_len = x.size(1)
            pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_encoding
            x = self.dropout(x)
            
            # Apply attention
            output, weights = self.attention(x, attn_mask=mask)
            
            return output, weights
            
        except Exception as e:
            logger.error(f"Error in temporal self-attention: {e}")
            return x, None
