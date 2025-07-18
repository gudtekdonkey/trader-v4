"""
Soft Actor-Critic (SAC) agent implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
import os
import json
import traceback

from .network_definitions import GaussianPolicy, QNetwork
from .replay_buffer import ReplayBuffer
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class SACAgent:
    """
    Soft Actor-Critic agent with comprehensive error handling.
    """
    
    def __init__(
        self, 
        state_size: int, 
        action_size: int,
        continuous_action_dim: int = 1,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: Optional[str] = None
    ):
        """Initialize SAC agent with validation."""
        # [ERROR-HANDLING] Validate inputs
        if state_size <= 0 or action_size <= 0:
            raise ValueError(f"Invalid sizes: state={state_size}, action={action_size}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.continuous_action_dim = continuous_action_dim
        
        # Device handling
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing SAC agent on device: {self.device}")
        
        # SAC parameters with validation
        self.gamma = np.clip(gamma, 0, 1)
        self.tau = np.clip(tau, 0, 1)
        self.alpha = alpha
        
        try:
            # Networks
            self.policy = GaussianPolicy(state_size, continuous_action_dim, hidden_dim).to(self.device)
            self.q_network = QNetwork(state_size, continuous_action_dim, hidden_dim).to(self.device)
            self.target_q_network = QNetwork(state_size, continuous_action_dim, hidden_dim).to(self.device)
            
            # Initialize target network
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            
            # Optimizers
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            
            # Automatic entropy tuning
            self.target_entropy = -torch.prod(torch.Tensor([continuous_action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            
        except Exception as e:
            logger.error(f"Error initializing networks: {e}")
            raise
        
        # Memory with error tracking
        self.memory = ReplayBuffer(capacity=100000, device=str(self.device))
        
        # Exploration parameters
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Training tracking
        self.training_steps = 0
        self.update_frequency = 1
        self.training_errors = 0
        self.max_training_errors = 100
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience with validation."""
        try:
            # [ERROR-HANDLING] Validate inputs
            if not isinstance(state, np.ndarray) or not isinstance(next_state, np.ndarray):
                logger.warning("Invalid state type")
                return
            
            if state.shape != (self.state_size,) or next_state.shape != (self.state_size,):
                logger.warning(f"Invalid state shape: {state.shape}, {next_state.shape}")
                return
            
            # Convert discrete action to continuous
            continuous_action = self._discrete_to_continuous(action)
            
            # Store in memory
            self.memory.push(state, continuous_action, reward, next_state, done)
            
        except Exception as e:
            logger.error(f"Error in remember: {e}")
    
    def act(self, state: np.ndarray) -> int:
        """Choose action with error handling."""
        try:
            # [ERROR-HANDLING] Validate state
            if not isinstance(state, np.ndarray) or state.shape != (self.state_size,):
                logger.warning(f"Invalid state for action: {state.shape if isinstance(state, np.ndarray) else type(state)}")
                return 0  # Default to hold
            
            # Epsilon-greedy exploration
            if np.random.random() <= self.epsilon:
                return np.random.choice(self.action_size)
            
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _, _ = self.policy.sample(state_tensor)
                continuous_action = action.cpu().data.numpy().flatten()
            
            # Convert to discrete
            return self._continuous_to_discrete(continuous_action)
            
        except Exception as e:
            logger.error(f"Error in act: {e}")
            return 0  # Default to hold
    
    def train(self, batch_size: int = 256) -> None:
        """Train agent with comprehensive error handling."""
        if len(self.memory) < batch_size:
            return
        
        try:
            # Sample batch
            state, action, reward, next_state, done = self.memory.sample(batch_size)
            
            # Update critic
            with torch.no_grad():
                next_action, next_log_prob, _ = self.policy.sample(next_state)
                target_q1, target_q2 = self.target_q_network(next_state, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                target_value = reward + (1 - done) * self.gamma * target_q
            
            current_q1, current_q2 = self.q_network(state, action)
            q1_loss = F.mse_loss(current_q1, target_value)
            q2_loss = F.mse_loss(current_q2, target_value)
            q_loss = q1_loss + q2_loss
            
            # [ERROR-HANDLING] Check for NaN loss
            if not torch.isfinite(q_loss):
                logger.warning("Non-finite Q loss detected")
                self.training_errors += 1
                return
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.q_optimizer.step()
            
            # Update actor
            if self.training_steps % self.update_frequency == 0:
                new_action, log_prob, _ = self.policy.sample(state)
                q1_new, q2_new = self.q_network(state, new_action)
                q_new = torch.min(q1_new, q2_new)
                
                policy_loss = (self.alpha * log_prob - q_new).mean()
                
                # [ERROR-HANDLING] Check for NaN loss
                if not torch.isfinite(policy_loss):
                    logger.warning("Non-finite policy loss detected")
                    self.training_errors += 1
                    return
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy_optimizer.step()
                
                # Update temperature
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp().item()
            
            # Soft update target networks
            if self.training_steps % self.update_frequency == 0:
                for param, target_param in zip(
                    self.q_network.parameters(), 
                    self.target_q_network.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.training_steps += 1
            
            # [ERROR-HANDLING] Check error threshold
            if self.training_errors > self.max_training_errors:
                logger.error("Too many training errors, stopping training")
                raise RuntimeError("Training instability detected")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            self.training_errors += 1
    
    def predict(self, state: np.ndarray) -> int:
        """Predict action without exploration (inference mode)"""
        return self.act(state)  # Can be extended for different inference behavior
    
    def get_confidence(self, state: np.ndarray) -> float:
        """Get confidence score for the action"""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, log_prob, _ = self.policy.sample(state_tensor)
                # Convert log probability to confidence (0-1 range)
                confidence = torch.exp(log_prob).item()
                return np.clip(confidence, 0, 1)
                
        except Exception as e:
            logger.error(f"Error getting confidence: {e}")
            return 0.5
    
    def _continuous_to_discrete(self, continuous_action: np.ndarray) -> int:
        """Convert continuous to discrete action with validation."""
        try:
            action_value = continuous_action[0] if len(continuous_action) > 0 else 0
            
            # [ERROR-HANDLING] Validate action value
            if not np.isfinite(action_value):
                logger.warning(f"Non-finite action value: {action_value}")
                return 0
            
            # Clip to valid range
            action_value = np.clip(action_value, -1, 1)
            
            # Map to discrete actions
            if -0.15 <= action_value <= 0.15:
                return 0  # Hold
            elif action_value > 0.15:
                if action_value <= 0.4:
                    return 1  # Buy 25%
                elif action_value <= 0.7:
                    return 2  # Buy 50%
                else:
                    return 3  # Buy 100%
            else:
                if action_value >= -0.4:
                    return 4  # Sell 25%
                elif action_value >= -0.7:
                    return 5  # Sell 50%
                else:
                    return 6  # Sell 100%
                    
        except Exception as e:
            logger.error(f"Error converting action: {e}")
            return 0
    
    def _discrete_to_continuous(self, discrete_action: int) -> np.ndarray:
        """Convert discrete to continuous action."""
        action_map = {
            0: 0.0, 1: 0.3, 2: 0.5, 3: 0.9,
            4: -0.3, 5: -0.5, 6: -0.9
        }
        
        continuous_value = action_map.get(discrete_action, 0.0)
        return np.array([continuous_value], dtype=np.float32)
    
    def save(self, filepath: str):
        """Save agent with error handling."""
        try:
            # [ERROR-HANDLING] Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save state dict
            state = {
                'policy_state_dict': self.policy.state_dict(),
                'q_network_state_dict': self.q_network.state_dict(),
                'target_q_network_state_dict': self.target_q_network.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'training_steps': self.training_steps,
                'epsilon': self.epsilon,
                'training_errors': self.training_errors
            }
            
            torch.save(state, filepath)
            logger.info(f"Saved SAC agent to {filepath}")
            
            # Also save metadata
            metadata_path = filepath.replace('.pt', '_metadata.json')
            metadata = {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'training_steps': self.training_steps,
                'training_errors': self.training_errors,
                'device': str(self.device)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving agent: {e}")
            raise
    
    def load(self, filepath: str):
        """Load agent with error handling."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load state dicts with error handling
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.training_steps = checkpoint.get('training_steps', 0)
            self.epsilon = checkpoint.get('epsilon', 0.01)
            self.training_errors = checkpoint.get('training_errors', 0)
            
            logger.info(f"Loaded SAC agent from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading agent: {e}")
            raise