import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class AttentionLayer(nn.Module):
    """Multi-head self-attention layer for LSTM"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear transformation
        output = self.fc_out(context)
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        return self.layer_norm(output + x)

class AttentionLSTM(nn.Module):
    """LSTM with multi-head self-attention for cryptocurrency price prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3,
                 output_dim: int = 1, num_heads: int = 8, dropout: float = 0.2):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(hidden_dim * 2, num_heads) for _ in range(2)
        ])
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        
        # Final predictions
        out = self.fc1(pooled[:, :self.hidden_dim * 2])
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        predictions = self.fc3(out)
        
        if return_attention:
            return predictions, attention_weights
        return predictions, None
    
    def predict_with_confidence(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions with confidence intervals using dropout uncertainty"""
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred, _ = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        
        self.eval()
        return mean_prediction, std_prediction
