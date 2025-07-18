"""
Gated Residual Network Module

Implements the GRN component that enables efficient information flow
with gating mechanism and skip connections, crucial for deep architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) component with error handling.
    
    GRN enables efficient information flow with gating mechanism
    and skip connections, crucial for deep architectures.
    
    Attributes:
        input_dim: Input dimension
        output_dim: Output dimension
        fc1, fc2: Linear transformation layers
        skip_layer: Optional skip connection for dimension matching
        gate: Gating mechanism layer
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        use_time_distributed: bool = True
    ):
        """
        Initialize the GRN with validation.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout rate
            use_time_distributed: Whether to use time-distributed layers
        """
        super().__init__()
        
        # Validate inputs
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("All dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {dropout}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_time_distributed = use_time_distributed
        
        try:
            # Main transformation layers
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            
            # Skip connection for dimension matching
            if input_dim != output_dim:
                self.skip_layer = nn.Linear(input_dim, output_dim)
            else:
                self.skip_layer = None
            
            self.dropout = nn.Dropout(dropout)
            self.gate = nn.Linear(input_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)
            
            # Initialize weights for stability
            self._init_weights()
            
        except Exception as e:
            logger.error(f"Failed to initialize GRN: {e}")
            raise
    
    def _init_weights(self):
        """Initialize weights for stability"""
        for module in [self.fc1, self.fc2, self.gate]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        if self.skip_layer is not None:
            nn.init.xavier_uniform_(self.skip_layer.weight)
            nn.init.constant_(self.skip_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GRN with error handling.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after gated residual connection
        """
        # Validate input
        if x is None:
            raise ValueError("Input tensor cannot be None")
        
        # Handle NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN/Inf in GRN input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        try:
            # Main pathway
            hidden = self.fc1(x)
            hidden = F.elu(hidden)
            hidden = self.dropout(hidden)
            hidden = self.fc2(hidden)
            
            # Skip connection
            if self.skip_layer is not None:
                skip = self.skip_layer(x)
            else:
                skip = x
            
            # Gating mechanism
            gating_layer = self.gate(x)
            
            # Stable sigmoid computation
            gate = torch.sigmoid(torch.clamp(gating_layer, min=-10, max=10))
            
            # Gated combination of main and skip pathways
            output = gate * hidden + (1 - gate) * skip
            
            # Check before layer norm
            if torch.isnan(output).any():
                logger.warning("NaN before layer norm in GRN")
                output = skip  # Use skip connection as fallback
            
            output = self.layer_norm(output)
            
            # Final check
            if torch.isnan(output).any():
                logger.warning("NaN in GRN output, using skip connection")
                return skip
            
            return output
            
        except Exception as e:
            logger.error(f"Error in GRN forward pass: {e}")
            # Return skip connection or input
            if self.skip_layer is not None:
                try:
                    return self.skip_layer(x)
                except:
                    return x
            return x
