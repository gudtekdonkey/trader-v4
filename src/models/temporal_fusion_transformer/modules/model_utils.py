"""
Model Utilities Module

Contains utility functions and validation helpers for the TFT model,
including input processing, loss calculations, and error handling.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class ModelUtils:
    """Utility functions for TFT model operations."""
    
    @staticmethod
    def process_input(x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Process various input formats into a single tensor.
        
        Args:
            x: Input data (dict or tensor)
            
        Returns:
            Processed tensor or None if processing fails
        """
        try:
            if isinstance(x, dict):
                if 'encoder_cont' in x:
                    return x['encoder_cont']
                elif 'features' in x:
                    return x['features']
                else:
                    # Concatenate all input tensors
                    valid_tensors = []
                    for k, v in x.items():
                        if isinstance(v, torch.Tensor):
                            if len(v.shape) == 3:  # Batch x Seq x Features
                                valid_tensors.append(v)
                            elif len(v.shape) == 2:  # Batch x Features
                                # Add sequence dimension
                                valid_tensors.append(v.unsqueeze(1))
                    
                    if not valid_tensors:
                        logger.error("No valid tensors found in input dictionary")
                        return None
                    
                    # Ensure compatible dimensions
                    max_seq_len = max(t.size(1) for t in valid_tensors)
                    aligned_tensors = []
                    for t in valid_tensors:
                        if t.size(1) < max_seq_len:
                            # Pad sequence dimension
                            padding = torch.zeros(
                                t.size(0), max_seq_len - t.size(1), t.size(2),
                                device=t.device, dtype=t.dtype
                            )
                            t = torch.cat([t, padding], dim=1)
                        aligned_tensors.append(t)
                    
                    return torch.cat(aligned_tensors, dim=-1)
            else:
                return x
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return None
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """
        Validate and clean tensor from NaN/Inf values.
        
        Args:
            tensor: Tensor to validate
            name: Name for logging
            
        Returns:
            Cleaned tensor
        """
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"NaN/Inf in {name}")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        return tensor
    
    @staticmethod
    def get_batch_size(x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> int:
        """Get batch size from input."""
        if isinstance(x, dict):
            for v in x.values():
                if isinstance(v, torch.Tensor):
                    return v.size(0)
            return 1
        else:
            return x.size(0)
    
    @staticmethod
    def get_device(x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.device:
        """Get device from input."""
        if isinstance(x, dict):
            for v in x.values():
                if isinstance(v, torch.Tensor):
                    return v.device
            return torch.device('cpu')
        else:
            return x.device
    
    @staticmethod
    def get_default_predictions(
        batch_size: int,
        device: torch.device,
        hidden_dim: int
    ) -> Dict[str, torch.Tensor]:
        """Get default predictions for error cases."""
        default_pred = torch.zeros((batch_size, 1), device=device)
        default_quantiles = torch.zeros((batch_size, 1, 3), device=device)
        default_encoder = torch.zeros((batch_size, hidden_dim), device=device)
        
        return {
            "prediction_outputs": default_pred,
            "quantiles": default_quantiles,
            "encoder_output": default_encoder,
            "attention_weights": None
        }
    
    @staticmethod
    def extract_batch_data(batch: Union[Dict, Tuple]) -> Tuple[Any, torch.Tensor]:
        """
        Extract inputs and targets from batch.
        
        Args:
            batch: Training batch
            
        Returns:
            Tuple of (inputs, targets)
        """
        if isinstance(batch, dict):
            x = {k: v for k, v in batch.items() if k != 'target'}
            y = batch['target']
        else:
            x, y = batch
        return x, y
    
    @staticmethod
    def calculate_quantile_loss(
        targets: torch.Tensor,
        predictions: torch.Tensor,
        quantile_levels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate quantile loss.
        
        Args:
            targets: Target values
            predictions: Predicted quantiles
            quantile_levels: Quantile levels
            
        Returns:
            Quantile loss
        """
        quantile_loss = 0
        for i, q in enumerate(quantile_levels):
            errors = targets.unsqueeze(-1) - predictions[..., i:i+1]
            quantile_loss += torch.mean(
                torch.max(q * errors, (q - 1) * errors)
            )
        return quantile_loss
    
    @staticmethod
    def init_weights(model: nn.Module) -> None:
        """
        Initialize model weights.
        
        Args:
            model: PyTorch model
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    @staticmethod
    def monte_carlo_predictions(
        model: nn.Module,
        x: Dict[str, torch.Tensor],
        num_samples: int = 100
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate predictions with uncertainty using Monte Carlo dropout.
        
        Args:
            model: TFT model
            x: Input features
            num_samples: Number of forward passes
            
        Returns:
            Dictionary with mean and std predictions
        """
        # Validate inputs
        if num_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Limit samples to prevent memory issues
        num_samples = min(num_samples, 500)
        
        model.train()  # Enable dropout for uncertainty estimation
        
        predictions_list = []
        
        try:
            with torch.no_grad():
                for i in range(num_samples):
                    try:
                        preds = model(x)
                        predictions_list.append(preds["prediction_outputs"])
                    except Exception as e:
                        logger.warning(f"Failed prediction {i}: {e}")
                        continue
            
            if not predictions_list:
                logger.error("All uncertainty predictions failed")
                # Return zero uncertainty
                batch_size = ModelUtils.get_batch_size(x)
                device = ModelUtils.get_device(x)
                return {
                    'price': (
                        torch.zeros((batch_size, 1), device=device),
                        torch.ones((batch_size, 1), device=device)
                    )
                }
            
            # Calculate statistics
            stacked = torch.stack(predictions_list)
            
            # Remove NaN predictions
            valid_mask = ~torch.isnan(stacked).any(dim=(1, 2))
            stacked = stacked[valid_mask]
            
            if len(stacked) == 0:
                logger.error("No valid predictions for uncertainty")
                batch_size = ModelUtils.get_batch_size(x)
                device = ModelUtils.get_device(x)
                return {
                    'price': (
                        torch.zeros((batch_size, 1), device=device),
                        torch.ones((batch_size, 1), device=device)
                    )
                }
            
            mean_pred = stacked.mean(dim=0)
            std_pred = stacked.std(dim=0)
            
            # Ensure std is not zero
            std_pred = torch.clamp(std_pred, min=1e-6)
            
            model.eval()
            
            return {
                'price': (mean_pred, std_pred)
            }
            
        except Exception as e:
            logger.error(f"Error in uncertainty estimation: {e}")
            model.eval()
            batch_size = ModelUtils.get_batch_size(x)
            device = ModelUtils.get_device(x)
            return {
                'price': (
                    torch.zeros((batch_size, 1), device=device),
                    torch.ones((batch_size, 1), device=device)
                )
            }


class TFTValidator:
    """Validation utilities for TFT model parameters."""
    
    @staticmethod
    def validate_model_params(
        input_dim: int,
        hidden_dim: int,
        lstm_layers: int,
        attention_heads: int,
        dropout: float,
        learning_rate: float
    ) -> None:
        """
        Validate TFT model parameters.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            lstm_layers: Number of LSTM layers
            attention_heads: Number of attention heads
            dropout: Dropout rate
            learning_rate: Learning rate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if lstm_layers <= 0:
            raise ValueError("Must have at least one LSTM layer")
        if attention_heads <= 0:
            raise ValueError("Must have at least one attention head")
        if not 0 <= dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {dropout}")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
    
    @staticmethod
    def validate_input_data(
        data: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> bool:
        """
        Validate input data format and content.
        
        Args:
            data: Input data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if isinstance(data, dict):
                if not data:
                    return False
                for k, v in data.items():
                    if not isinstance(v, torch.Tensor):
                        return False
                    if v.numel() == 0:
                        return False
            elif isinstance(data, torch.Tensor):
                if data.numel() == 0:
                    return False
            else:
                return False
            return True
        except Exception:
            return False
