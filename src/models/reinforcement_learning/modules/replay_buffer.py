"""
Experience replay buffer for reinforcement learning
"""

import numpy as np
import torch
from collections import deque
import random
from typing import Tuple
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class ReplayBuffer:
    """Optimized experience replay buffer with error handling."""
    
    def __init__(self, capacity: int = 100000, device: str = "cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.push_errors = 0
        self.sample_errors = 0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience with validation"""
        try:
            # [ERROR-HANDLING] Validate inputs
            if not all(isinstance(x, np.ndarray) for x in [state, next_state]):
                logger.warning("Invalid state type for replay buffer")
                return
            
            if not all(np.isfinite(x).all() for x in [state, next_state]):
                logger.warning("Non-finite values in states")
                return
            
            if not isinstance(reward, (int, float)) or not np.isfinite(reward):
                logger.warning(f"Invalid reward: {reward}")
                return
            
            self.buffer.append((state, action, reward, next_state, done))
            
        except Exception as e:
            logger.error(f"Error pushing to replay buffer: {e}")
            self.push_errors += 1
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with error handling"""
        try:
            if len(self.buffer) < batch_size:
                raise ValueError(f"Not enough samples: {len(self.buffer)} < {batch_size}")
            
            batch = random.sample(self.buffer, batch_size)
            
            # [ERROR-HANDLING] Validate batch
            valid_batch = []
            for experience in batch:
                if len(experience) == 5 and all(x is not None for x in experience):
                    valid_batch.append(experience)
            
            if len(valid_batch) < batch_size:
                logger.warning(f"Some invalid experiences filtered: {len(valid_batch)}/{batch_size}")
                if len(valid_batch) < batch_size // 2:
                    raise ValueError("Too many invalid experiences")
            
            state, action, reward, next_state, done = map(np.stack, zip(*valid_batch))
            
            # Convert to tensors with validation
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
            
            # [ERROR-HANDLING] Final validation
            tensors = [state, action, reward, next_state, done]
            for i, tensor in enumerate(tensors):
                if not torch.isfinite(tensor).all():
                    logger.warning(f"Non-finite values in tensor {i}")
                    # Replace with zeros
                    tensors[i] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return tuple(tensors)
            
        except Exception as e:
            logger.error(f"Error sampling from replay buffer: {e}")
            self.sample_errors += 1
            raise
    
    def __len__(self):
        return len(self.buffer)