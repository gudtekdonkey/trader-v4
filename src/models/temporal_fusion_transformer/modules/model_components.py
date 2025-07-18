"""
Model Components Module

Contains various components used in the TFT model including
temporal encoder, quantile heads, and input embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class InputEmbedding(nn.Module):
    """
    Input embedding layer for TFT.
    
    Handles the initial transformation of input features
    to the model's hidden dimension.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize input embedding.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply input embedding"""
        try:
            embedded = self.linear(x)
            return self.activation(embedded)
        except Exception as e:
            logger.error(f"Error in input embedding: {e}")
            return x


class TemporalEncoder(nn.Module):
    """
    Temporal encoder using LSTM for sequence processing.
    
    Processes temporal sequences with bidirectional LSTM
    and returns encoded representations.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        """
        Initialize temporal encoder.
        
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("Hidden dim and num_layers must be positive")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode temporal sequence.
        
        Args:
            x: Input tensor
            batch_size: Batch size
            device: Device to use
            
        Returns:
            Tuple of (output, (hidden, cell))
        """
        try:
            # Initialize hidden states
            h0 = torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_dim, 
                device=device
            )
            c0 = torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_dim, 
                device=device
            )
            
            output, (hidden, cell) = self.lstm(x, (h0, c0))
            
            return output, (hidden, cell)
            
        except Exception as e:
            logger.error(f"Error in temporal encoder: {e}")
            # Return input repeated for bidirectional
            return x.repeat(1, 1, 2), (h0, c0)


class QuantileHeads(nn.Module):
    """
    Quantile prediction heads for uncertainty estimation.
    
    Generates predictions at multiple quantile levels
    (10%, 50%, 90%) for uncertainty quantification.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_size: int,
        quantiles: List[float] = None
    ):
        """
        Initialize quantile heads.
        
        Args:
            hidden_dim: Hidden dimension
            output_size: Output dimension
            quantiles: List of quantile levels
        """
        super().__init__()
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.quantiles = quantiles
        self.output_size = output_size
        
        # Create a head for each quantile
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_size)
            for _ in quantiles
        ])
        
        # Initialize weights
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate quantile predictions.
        
        Args:
            x: Input features
            
        Returns:
            Tensor of shape (batch, output_size, num_quantiles)
        """
        try:
            quantiles = []
            
            for head in self.heads:
                try:
                    q_pred = head(x)
                    quantiles.append(q_pred)
                except Exception as e:
                    logger.error(f"Error in quantile head: {e}")
                    # Use zeros as fallback
                    quantiles.append(torch.zeros(
                        x.size(0), self.output_size, device=x.device
                    ))
            
            # Stack and sort quantiles
            quantiles = torch.stack(quantiles, dim=-1)
            quantiles, _ = torch.sort(quantiles, dim=-1)
            
            return quantiles
            
        except Exception as e:
            logger.error(f"Error in quantile prediction: {e}")
            # Return default quantiles
            return torch.zeros(
                x.size(0), self.output_size, len(self.quantiles), 
                device=x.device
            )


class StaticCovariateEncoder(nn.Module):
    """
    Encoder for static covariates that don't change over time.
    
    Processes static features and creates context vectors
    that modulate the temporal processing.
    """
    
    def __init__(
        self,
        static_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize static covariate encoder.
        
        Args:
            static_dim: Dimension of static features
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Context vectors for different components
        self.variable_selection_context = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_context = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_context = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, static_features: torch.Tensor) -> dict:
        """
        Encode static features into context vectors.
        
        Args:
            static_features: Static feature tensor
            
        Returns:
            Dictionary of context vectors
        """
        try:
            encoded = self.encoder(static_features)
            
            return {
                'variable_selection': self.variable_selection_context(encoded),
                'encoder': self.encoder_context(encoded),
                'decoder': self.decoder_context(encoded)
            }
            
        except Exception as e:
            logger.error(f"Error in static covariate encoder: {e}")
            # Return zero contexts
            batch_size = static_features.size(0)
            hidden_dim = self.variable_selection_context.out_features
            zero_context = torch.zeros(batch_size, hidden_dim, device=static_features.device)
            
            return {
                'variable_selection': zero_context,
                'encoder': zero_context,
                'decoder': zero_context
            }
