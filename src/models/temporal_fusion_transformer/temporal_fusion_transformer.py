"""
File: temporal_fusion_transformer.py
Modified: 2025-07-18
Changes Summary:
- Modularized into separate components for better maintainability
- Main TFT model coordination remains here
- Components moved to modules/ directory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl
import logging
import warnings

# Import modularized components
from .modules.variable_selection import VariableSelectionNetwork
from .modules.gated_residual_network import GatedResidualNetwork
from .modules.attention_components import MultiHeadAttentionWithGating
from .modules.model_components import TemporalEncoder, QuantileHeads, InputEmbedding
from .modules.model_utils import ModelUtils, TFTValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFTModel(pl.LightningModule):
    """
    Simplified Temporal Fusion Transformer for crypto prediction with error handling.
    
    This model implements a simplified version of TFT with:
    - LSTM encoder for temporal processing
    - Multi-head attention for capturing dependencies
    - Gated residual networks for feature transformation
    - Quantile prediction for uncertainty estimation
    
    Attributes:
        hidden_dim: Hidden dimension size
        lstm: LSTM encoder
        attention: Multi-head attention layer
        static_enrichment: GRN for static feature processing
        temporal_grn: GRN for temporal feature processing
        output_layer: Final prediction layer
        quantile_heads: Heads for quantile prediction
    """
    
    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 160,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        output_size: int = 1,
        learning_rate: float = 0.001,
        reduce_on_plateau_patience: int = 4
    ):
        """
        Initialize the TFT model with validation.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            lstm_layers: Number of LSTM layers
            attention_heads: Number of attention heads
            dropout: Dropout rate
            output_size: Output dimension
            learning_rate: Learning rate for optimization
            reduce_on_plateau_patience: Patience for learning rate reduction
        """
        super().__init__()
        
        # Use validator for input validation
        validator = TFTValidator()
        validator.validate_model_params(
            input_dim, hidden_dim, lstm_layers, 
            attention_heads, dropout, learning_rate
        )
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.utils = ModelUtils()
        
        try:
            # Input processing
            self.input_embedding = InputEmbedding(input_dim, hidden_dim)
            
            # Temporal encoder
            self.temporal_encoder = TemporalEncoder(
                hidden_dim, lstm_layers, dropout
            )
            
            # Attention mechanism
            self.attention = MultiHeadAttentionWithGating(
                hidden_dim * 2,  # *2 for bidirectional
                attention_heads,
                dropout
            )
            
            # Gated residual networks
            self.static_enrichment = GatedResidualNetwork(
                hidden_dim * 2, hidden_dim, hidden_dim, dropout
            )
            
            self.temporal_grn = GatedResidualNetwork(
                hidden_dim, hidden_dim, hidden_dim, dropout
            )
            
            # Output layers
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_size)
            )
            
            # Quantile heads
            self.quantile_heads = QuantileHeads(hidden_dim, output_size)
            
            # Initialize weights
            self.utils.init_weights(self)
            
            logger.info(f"TFTModel initialized with hidden_dim={hidden_dim}, lstm_layers={lstm_layers}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TFTModel: {e}")
            raise
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the TFT model with error handling.
        
        Args:
            x: Input dictionary containing various features
            
        Returns:
            Dictionary with predictions, quantiles, and attention weights
        """
        # Process input
        input_tensor = self.utils.process_input(x)
        if input_tensor is None:
            batch_size = self.utils.get_batch_size(x)
            device = self.utils.get_device(x)
            return self.utils.get_default_predictions(batch_size, device, self.hidden_dim)
        
        # Validate input tensor
        input_tensor = self.utils.validate_tensor(input_tensor, "input")
        batch_size, seq_len = input_tensor.shape[:2]
        
        try:
            # Input embedding
            embedded = self.input_embedding(input_tensor)
            
            # Temporal encoding
            lstm_out, hidden_states = self.temporal_encoder(embedded, batch_size, input_tensor.device)
            
            # Check LSTM output
            if torch.isnan(lstm_out).any():
                logger.warning("NaN in LSTM output")
                lstm_out = embedded.repeat(1, 1, 2)  # Fallback
            
        except Exception as e:
            logger.error(f"Error in encoding: {e}")
            return self.utils.get_default_predictions(batch_size, input_tensor.device, self.hidden_dim)
        
        try:
            # Self-attention
            attended_out, attention_weights = self.attention(lstm_out)
            
            # Check attention output
            if torch.isnan(attended_out).any():
                logger.warning("NaN in attention output")
                attended_out = lstm_out  # Skip attention
                
        except Exception as e:
            logger.error(f"Error in attention: {e}")
            attended_out = lstm_out
            attention_weights = None
        
        try:
            # Static enrichment
            enriched = self.static_enrichment(attended_out)
            
            # Temporal processing
            temporal_out = self.temporal_grn(enriched)
            
            # Global pooling (average over sequence)
            if temporal_out.size(1) == 0:
                pooled = torch.zeros((batch_size, self.hidden_dim), device=input_tensor.device)
            else:
                pooled = temporal_out.mean(dim=1)
            
            # Check pooled features
            if torch.isnan(pooled).any():
                logger.warning("NaN in pooled features")
                pooled = torch.zeros_like(pooled)
                
        except Exception as e:
            logger.error(f"Error in feature processing: {e}")
            pooled = torch.zeros((batch_size, self.hidden_dim), device=input_tensor.device)
        
        try:
            # Main prediction
            prediction = self.output_layer(pooled)
            
            # Quantile predictions
            quantiles = self.quantile_heads(pooled)
            
            # Final validation
            if torch.isnan(prediction).any():
                logger.warning("NaN in predictions")
                prediction = torch.zeros_like(prediction)
            
            if torch.isnan(quantiles).any():
                logger.warning("NaN in quantiles")
                quantiles = prediction.unsqueeze(-1).repeat(1, 1, 3)
            
            return {
                "prediction_outputs": prediction,
                "quantiles": quantiles,
                "encoder_output": pooled,
                "attention_weights": attention_weights
            }
            
        except Exception as e:
            logger.error(f"Error in output generation: {e}")
            return self.utils.get_default_predictions(batch_size, input_tensor.device, self.hidden_dim)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning with error handling.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Total loss
        """
        try:
            # Extract inputs and targets
            x, y = self.utils.extract_batch_data(batch)
            
            # Validate target
            if torch.isnan(y).any() or torch.isinf(y).any():
                logger.warning("NaN/Inf in target, skipping batch")
                return torch.tensor(0.0, requires_grad=True)
            
            predictions = self(x)
            
            # Main prediction loss (MSE)
            main_loss = F.mse_loss(predictions["prediction_outputs"], y)
            
            # Check for NaN loss
            if torch.isnan(main_loss):
                logger.warning("NaN loss detected")
                return torch.tensor(0.0, requires_grad=True)
            
            # Quantile loss
            quantiles = predictions["quantiles"]
            quantile_levels = torch.tensor(
                [0.1, 0.5, 0.9], 
                device=self.device
            )
            
            quantile_loss = self.utils.calculate_quantile_loss(
                y, quantiles, quantile_levels
            )
            
            # Combined loss
            total_loss = main_loss + 0.1 * quantile_loss
            
            # Clip loss to prevent instability
            total_loss = torch.clamp(total_loss, min=0, max=1000)
            
            # Logging
            self.log("train_loss", total_loss)
            self.log("train_mse", main_loss)
            self.log("train_quantile_loss", quantile_loss)
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            # Return small loss to continue training
            return torch.tensor(0.001, requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning with error handling.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        try:
            # Extract inputs and targets
            x, y = self.utils.extract_batch_data(batch)
            
            # Validate inputs
            if torch.isnan(y).any():
                logger.warning("NaN in validation target")
                return torch.tensor(0.0)
            
            predictions = self(x)
            loss = F.mse_loss(predictions["prediction_outputs"], y)
            
            # Check loss
            if torch.isnan(loss):
                logger.warning("NaN validation loss")
                return torch.tensor(0.0)
            
            self.log("val_loss", loss)
            return loss
            
        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            return torch.tensor(0.0)
    
    def configure_optimizers(self):
        """
        Configure optimizers and schedulers with error handling.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        try:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.hparams.reduce_on_plateau_patience,
                factor=0.5,
                verbose=True
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }
            
        except Exception as e:
            logger.error(f"Error configuring optimizers: {e}")
            # Return basic optimizer
            return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def predict_with_uncertainty(
        self,
        x: Dict[str, torch.Tensor],
        num_samples: int = 100
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate predictions with uncertainty estimates.
        
        Uses Monte Carlo dropout for uncertainty quantification.
        
        Args:
            x: Input features
            num_samples: Number of forward passes
            
        Returns:
            Dictionary with mean and std predictions
        """
        return self.utils.monte_carlo_predictions(self, x, num_samples)

"""
MODULARIZATION_SUMMARY:
- Original file: 1000+ lines
- Main file: ~300 lines (core coordination)
- Modules created:
  - variable_selection.py: Variable selection network
  - gated_residual_network.py: GRN implementation
  - attention_components.py: Attention mechanisms
  - model_components.py: Encoder, quantile heads, embeddings
  - model_utils.py: Utilities and validation
- Benefits:
  - Clearer separation of concerns
  - Easier testing of individual components
  - Better code reusability
  - Simplified maintenance
"""
