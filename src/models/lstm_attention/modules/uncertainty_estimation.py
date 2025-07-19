"""Uncertainty estimation utilities for LSTM models"""

import torch
import torch.nn as nn
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """Handles uncertainty estimation using Monte Carlo dropout"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize uncertainty estimator.
        
        Args:
            model: The neural network model to estimate uncertainty for
        """
        self.model = model
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions with confidence intervals using dropout uncertainty.
        
        This method uses Monte Carlo dropout to estimate prediction uncertainty
        by running multiple forward passes with dropout enabled.
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        # Input validation
        if x is None:
            raise ValueError("Input tensor cannot be None")
        if num_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {num_samples}")
        
        # Limit number of samples to prevent memory issues
        num_samples = min(num_samples, 1000)
        if num_samples > 200:
            logger.warning(f"Large number of samples {num_samples} may cause memory issues")
        
        # Store original training mode
        was_training = self.model.training
        self.model.train()  # Enable dropout for uncertainty estimation
        
        predictions = []
        
        try:
            with torch.no_grad():
                for i in range(num_samples):
                    try:
                        pred, _ = self.model(x)
                        predictions.append(pred)
                    except Exception as e:
                        logger.warning(f"Failed prediction {i}: {e}")
                        continue
            
            if not predictions:
                logger.error("All uncertainty predictions failed")
                # Return zero uncertainty
                return torch.zeros_like(x[:, 0, 0:1]), torch.ones_like(x[:, 0, 0:1])
            
            # Stack predictions and calculate statistics
            predictions = torch.stack(predictions)
            
            # Remove any NaN predictions
            valid_mask = ~torch.isnan(predictions).any(dim=(1, 2))
            predictions = predictions[valid_mask]
            
            if len(predictions) == 0:
                logger.error("No valid predictions for uncertainty estimation")
                return torch.zeros_like(x[:, 0, 0:1]), torch.ones_like(x[:, 0, 0:1])
            
            mean_prediction = predictions.mean(dim=0)
            std_prediction = predictions.std(dim=0)
            
            # Ensure std is not zero
            std_prediction = torch.clamp(std_prediction, min=1e-6)
            
        except Exception as e:
            logger.error(f"Error in uncertainty estimation: {e}")
            mean_prediction = torch.zeros_like(x[:, 0, 0:1])
            std_prediction = torch.ones_like(x[:, 0, 0:1])
        
        finally:
            # Restore original training mode
            self.model.train(was_training)
        
        return mean_prediction, std_prediction
    
    def get_epistemic_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 50
    ) -> torch.Tensor:
        """
        Calculate epistemic (model) uncertainty.
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes
            
        Returns:
            Epistemic uncertainty measure
        """
        _, std_prediction = self.predict_with_uncertainty(x, num_samples)
        
        # Epistemic uncertainty is captured by the standard deviation
        # of predictions across multiple dropout samples
        epistemic_uncertainty = std_prediction.mean()
        
        return epistemic_uncertainty
    
    def get_prediction_intervals(
        self,
        x: torch.Tensor,
        num_samples: int = 100,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prediction intervals at specified confidence level.
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes
            confidence_level: Confidence level (e.g., 0.95 for 95% intervals)
            
        Returns:
            Tuple of (mean_prediction, lower_bound, upper_bound)
        """
        mean_pred, std_pred = self.predict_with_uncertainty(x, num_samples)
        
        # Calculate z-score for confidence level
        # For 95% confidence: z = 1.96
        if confidence_level == 0.95:
            z_score = 1.96
        elif confidence_level == 0.99:
            z_score = 2.576
        elif confidence_level == 0.90:
            z_score = 1.645
        else:
            # Use normal distribution quantile
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate prediction intervals
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        return mean_pred, lower_bound, upper_bound
