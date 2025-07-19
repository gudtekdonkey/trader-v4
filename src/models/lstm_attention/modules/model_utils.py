"""Model utilities and helper functions"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates model inputs and outputs"""
    
    @staticmethod
    def validate_input(x: torch.Tensor, expected_features: Optional[int] = None) -> None:
        """
        Validate model input tensor.
        
        Args:
            x: Input tensor to validate
            expected_features: Expected number of features (optional)
            
        Raises:
            ValueError: If input is invalid
        """
        if x is None:
            raise ValueError("Input tensor cannot be None")
        
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D tensor (batch, seq_len, features), got shape {x.shape}")
        
        if x.shape[0] == 0:
            raise ValueError("Batch size cannot be zero")
        
        if x.shape[1] == 0:
            raise ValueError("Sequence length cannot be zero")
        
        if expected_features is not None and x.shape[2] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {x.shape[2]}")
        
        # Check for NaN or Inf
        if torch.isnan(x).any():
            logger.warning("Input contains NaN values")
        
        if torch.isinf(x).any():
            logger.warning("Input contains Inf values")
    
    @staticmethod
    def validate_output(output: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Validate and sanitize model output.
        
        Args:
            output: Output tensor to validate
            batch_size: Expected batch size
            
        Returns:
            Sanitized output tensor
        """
        if output is None:
            logger.error("Output is None, returning zeros")
            return torch.zeros((batch_size, 1))
        
        # Check for NaN or Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.warning("Output contains NaN or Inf, clamping values")
            output = torch.nan_to_num(output, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Clamp to reasonable range
        output = torch.clamp(output, min=-100, max=100)
        
        return output


class ModelCheckpointer:
    """Handles model checkpointing and recovery"""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer = None):
        """
        Initialize checkpointer.
        
        Args:
            model: Model to checkpoint
            optimizer: Optional optimizer to include in checkpoint
        """
        self.model = model
        self.optimizer = optimizer
        self.best_loss = float('inf')
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        path: str
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            metrics: Additional metrics to save
            path: Path to save checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'metrics': metrics
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        try:
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path}")
            
            # Track checkpoint history
            self.checkpoint_history.append({
                'epoch': epoch,
                'loss': loss,
                'path': path
            })
            
            # Update best loss
            if loss < self.best_loss:
                self.best_loss = loss
                logger.info(f"New best loss: {loss:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available
            if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"Loaded checkpoint from {path}")
            logger.info(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


class GradientClipper:
    """Handles gradient clipping for stable training"""
    
    def __init__(self, max_norm: float = 1.0):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum gradient norm
        """
        self.max_norm = max_norm
        self.clip_history = []
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients of model parameters.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Total gradient norm before clipping
        """
        total_norm = 0.0
        
        # Calculate total gradient norm
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        
        # Track clipping history
        self.clip_history.append(total_norm)
        
        if total_norm > self.max_norm * 2:
            logger.warning(f"Large gradient norm: {total_norm:.2f}")
        
        return total_norm
    
    def get_statistics(self) -> Dict[str, float]:
        """Get gradient clipping statistics"""
        if not self.clip_history:
            return {}
        
        return {
            'mean_grad_norm': np.mean(self.clip_history),
            'max_grad_norm': np.max(self.clip_history),
            'min_grad_norm': np.min(self.clip_history),
            'clip_rate': sum(1 for x in self.clip_history if x > self.max_norm) / len(self.clip_history)
        }
