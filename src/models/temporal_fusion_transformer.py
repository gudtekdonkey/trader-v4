import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl

class VariableSelectionNetwork(nn.Module):
    """Variable selection network for TFT"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
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
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sparse_weights = self.flattened_grn(x)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        
        trans_emb_list = []
        for i in range(self.output_dim):
            e = self.grns[i](x[..., i:i+1])
            trans_emb_list.append(e)
            
        transformed_embedding = torch.cat(trans_emb_list, dim=-1)
        
        combined = torch.sum(sparse_weights * transformed_embedding, dim=1)
        
        return combined, sparse_weights

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network component"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout: float = 0.1, use_time_distributed: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_time_distributed = use_time_distributed
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        if input_dim != output_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = None
            
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x
            
        gating_layer = self.gate(x)
        gate = torch.sigmoid(gating_layer)
        
        output = gate * hidden + (1 - gate) * skip
        output = self.layer_norm(output)
        
        return output

class TFTModel(pl.LightningModule):
    """Simplified Temporal Fusion Transformer for crypto prediction"""
    
    def __init__(self, 
                 input_dim: int = 100,
                 hidden_dim: int = 160,
                 lstm_layers: int = 2,
                 attention_heads: int = 4,
                 dropout: float = 0.1,
                 output_size: int = 1,
                 learning_rate: float = 0.001,
                 reduce_on_plateau_patience: int = 4):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        
        # Input processing
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM encoder
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
            hidden_dim * 2, 
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
        
        # Quantile heads for uncertainty
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_size) for _ in [0.1, 0.5, 0.9]
        ])
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Handle different input formats
        if isinstance(x, dict):
            if 'encoder_cont' in x:
                input_tensor = x['encoder_cont']
            elif 'features' in x:
                input_tensor = x['features']
            else:
                # Concatenate all input tensors
                input_tensor = torch.cat([v for v in x.values() if isinstance(v, torch.Tensor)], dim=-1)
        else:
            input_tensor = x
        
        batch_size, seq_len = input_tensor.shape[:2]
        
        # Input embedding
        embedded = self.input_embedding(input_tensor)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Static enrichment
        enriched = self.static_enrichment(attended_out)
        
        # Temporal processing
        temporal_out = self.temporal_grn(enriched)
        
        # Global pooling
        pooled = temporal_out.mean(dim=1)  # Average pooling over sequence
        
        # Main prediction
        prediction = self.output_layer(pooled)
        
        # Quantile predictions
        quantiles = torch.stack([head(pooled) for head in self.quantile_heads], dim=-1)
        
        return {
            "prediction_outputs": prediction,
            "quantiles": quantiles,
            "encoder_output": pooled,
            "attention_weights": attention_weights
        }
    
    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = {k: v for k, v in batch.items() if k != 'target'}
            y = batch['target']
        else:
            x, y = batch
        
        predictions = self(x)
        
        # Main loss
        main_loss = F.mse_loss(predictions["prediction_outputs"], y)
        
        # Quantile loss
        quantiles = predictions["quantiles"]
        quantile_levels = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        
        quantile_loss = 0
        for i, q in enumerate(quantile_levels):
            errors = y.unsqueeze(-1) - quantiles[..., i:i+1]
            quantile_loss += torch.mean(torch.max(q * errors, (q - 1) * errors))
        
        total_loss = main_loss + 0.1 * quantile_loss
        
        self.log("train_loss", total_loss)
        self.log("train_mse", main_loss)
        self.log("train_quantile_loss", quantile_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.hparams.reduce_on_plateau_patience, factor=0.5, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def predict_with_uncertainty(self, x: Dict[str, torch.Tensor], num_samples: int = 100) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate predictions with uncertainty estimates"""
        self.train()  # Enable dropout
        
        predictions_list = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                preds = self(x)
                predictions_list.append(preds["prediction_outputs"])
        
        # Calculate mean and std
        stacked = torch.stack(predictions_list)
        mean_pred = stacked.mean(dim=0)
        std_pred = stacked.std(dim=0)
        
        self.eval()
        
        return {
            'price': (mean_pred, std_pred)
        }
