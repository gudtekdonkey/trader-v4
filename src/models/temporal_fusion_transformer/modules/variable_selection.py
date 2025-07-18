"""
Variable Selection Network Module

Implements the variable selection mechanism for TFT,
which learns to select and weight different input variables
dynamically based on their importance for prediction.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple

from .gated_residual_network import GatedResidualNetwork

logger = logging.getLogger(__name__)


class VariableSelectionNetwork(nn.Module):
    """
    Variable selection network for feature importance in TFT with error handling.
    
    This network learns to select and weight different input variables
    dynamically based on their importance for prediction.
    
    Attributes:
        input_dim: Dimension of input features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize the variable selection network with validation.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer size
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Validate inputs
        if input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"Output dimension must be positive, got {output_dim}")
        if not 0 <= dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {dropout}")
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        try:
            self.flattened_grn = GatedResidualNetwork(
                input_dim,
                hidden_dim,
                output_dim,
                dropout
            )
            
            self.softmax = nn.Softmax(dim=-1)
            
            # Individual GRNs for each variable
            self.grns = nn.ModuleList([
                GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
                for _ in range(output_dim)
            ])
            
            logger.info(f"VariableSelectionNetwork initialized with output_dim={output_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VariableSelectionNetwork: {e}")
            raise
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for variable selection with error handling.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (selected features, selection weights)
        """
        # Validate input
        if x is None:
            raise ValueError("Input tensor cannot be None")
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN/Inf in variable selection input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        try:
            # Get selection weights
            sparse_weights = self.flattened_grn(x)
            
            # Check for NaN in weights
            if torch.isnan(sparse_weights).any():
                logger.warning("NaN in selection weights, using uniform weights")
                sparse_weights = torch.ones_like(sparse_weights) / self.output_dim
            else:
                sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
            
            # Transform each variable independently
            trans_emb_list = []
            for i in range(self.output_dim):
                try:
                    # Safe indexing
                    if i < x.size(-1):
                        e = self.grns[i](x[..., i:i+1])
                    else:
                        logger.warning(f"Variable index {i} out of bounds, using zeros")
                        e = torch.zeros((*x.shape[:-1], self.hidden_dim), device=x.device)
                    trans_emb_list.append(e)
                except Exception as e:
                    logger.error(f"Error processing variable {i}: {e}")
                    # Use zeros for failed variable
                    trans_emb_list.append(torch.zeros((*x.shape[:-1], self.hidden_dim), device=x.device))
            
            # Ensure we have the right number of embeddings
            while len(trans_emb_list) < self.output_dim:
                trans_emb_list.append(torch.zeros((*x.shape[:-1], self.hidden_dim), device=x.device))
            
            transformed_embedding = torch.cat(trans_emb_list, dim=-1)
            
            # Apply weighted selection
            combined = torch.sum(sparse_weights * transformed_embedding, dim=1)
            
            # Final check
            if torch.isnan(combined).any():
                logger.warning("NaN in variable selection output")
                combined = torch.zeros_like(combined)
            
            return combined, sparse_weights
            
        except Exception as e:
            logger.error(f"Critical error in variable selection: {e}")
            # Return safe defaults
            batch_size = x.size(0)
            combined = torch.zeros((batch_size, self.hidden_dim), device=x.device)
            weights = torch.ones((batch_size, self.output_dim, 1), device=x.device) / self.output_dim
            return combined, weights
