"""
Temporal Fusion Transformer (TFT) for Cryptocurrency Price Prediction

This module implements a simplified version of the Temporal Fusion Transformer
architecture with:
- Variable selection networks
- Gated residual networks
- Multi-head attention mechanism
- Quantile prediction for uncertainty estimation
- PyTorch Lightning integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl


class VariableSelectionNetwork(nn.Module):
    """
    Variable selection network for feature importance in TFT.
    
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
        Initialize the variable selection network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer size
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for variable selection.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (selected features, selection weights)
        """
        # Get selection weights
        sparse_weights = self.flattened_grn(x)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        
        # Transform each variable independently
        trans_emb_list = []
        for i in range(self.output_dim):
            e = self.grns[i](x[..., i:i+1])
            trans_emb_list.append(e)
        
        transformed_embedding = torch.cat(trans_emb_list, dim=-1)
        
        # Apply weighted selection
        combined = torch.sum(sparse_weights * transformed_embedding, dim=1)
        
        return combined, sparse_weights


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) component.
    
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
        Initialize the GRN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout rate
            use_time_distributed: Whether to use time-distributed layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_time_distributed = use_time_distributed
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GRN.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after gated residual connection
        """
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
        gate = torch.sigmoid(gating_layer)
        
        # Gated combination of main and skip pathways
        output = gate * hidden + (1 - gate) * skip
        output = self.layer_norm(output)
        
        return output


class TFTModel(pl.LightningModule):
    """
    Simplified Temporal Fusion Transformer for crypto prediction.
    
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
        Initialize the TFT model.
        
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
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        
        # Input processing
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM encoder (bidirectional)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # *2 for bidirectional
            attention_heads,
            dropout=dropout,
            batch_first=True
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
        
        # Quantile heads for uncertainty (10%, 50%, 90%)
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_size) 
            for _ in [0.1, 0.5, 0.9]
        ])
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the TFT model.
        
        Args:
            x: Input dictionary containing various features
            
        Returns:
            Dictionary with predictions, quantiles, and attention weights
        """
        # Handle different input formats
        if isinstance(x, dict):
            if 'encoder_cont' in x:
                input_tensor = x['encoder_cont']
            elif 'features' in x:
                input_tensor = x['features']
            else:
                # Concatenate all input tensors
                input_tensor = torch.cat([
                    v for v in x.values() 
                    if isinstance(v, torch.Tensor)
                ], dim=-1)
        else:
            input_tensor = x
        
        batch_size, seq_len = input_tensor.shape[:2]
        
        # Input embedding
        embedded = self.input_embedding(input_tensor)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Static enrichment
        enriched = self.static_enrichment(attended_out)
        
        # Temporal processing
        temporal_out = self.temporal_grn(enriched)
        
        # Global pooling (average over sequence)
        pooled = temporal_out.mean(dim=1)
        
        # Main prediction
        prediction = self.output_layer(pooled)
        
        # Quantile predictions
        quantiles = torch.stack([
            head(pooled) for head in self.quantile_heads
        ], dim=-1)
        
        return {
            "prediction_outputs": prediction,
            "quantiles": quantiles,
            "encoder_output": pooled,
            "attention_weights": attention_weights
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Total loss
        """
        # Extract inputs and targets
        if isinstance(batch, dict):
            x = {k: v for k, v in batch.items() if k != 'target'}
            y = batch['target']
        else:
            x, y = batch
        
        predictions = self(x)
        
        # Main prediction loss (MSE)
        main_loss = F.mse_loss(predictions["prediction_outputs"], y)
        
        # Quantile loss
        quantiles = predictions["quantiles"]
        quantile_levels = torch.tensor(
            [0.1, 0.5, 0.9], 
            device=self.device
        )
        
        quantile_loss = 0
        for i, q in enumerate(quantile_levels):
            errors = y.unsqueeze(-1) - quantiles[..., i:i+1]
            quantile_loss += torch.mean(
                torch.max(q * errors, (q - 1) * errors)
            )
        
        # Combined loss
        total_loss = main_loss + 0.1 * quantile_loss
        
        # Logging
        self.log("train_loss", total_loss)
        self.log("train_mse", main_loss)
        self.log("train_quantile_loss", quantile_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        # Extract inputs and targets
        if isinstance(batch, dict):
            x = {k: v for k, v in batch.items() if k != 'target'}
            y = batch['target']
        else:
            x, y = batch
        
        predictions = self(x)
        loss = F.mse_loss(predictions["prediction_outputs"], y)
        
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
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
        self.train()  # Enable dropout for uncertainty estimation
        
        predictions_list = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                preds = self(x)
                predictions_list.append(preds["prediction_outputs"])
        
        # Calculate statistics
        stacked = torch.stack(predictions_list)
        mean_pred = stacked.mean(dim=0)
        std_pred = stacked.std(dim=0)
        
        self.eval()
        
        return {
            'price': (mean_pred, std_pred)
        }