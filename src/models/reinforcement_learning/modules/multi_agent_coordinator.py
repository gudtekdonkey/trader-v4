"""
Multi-agent trading system coordinator
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional
import os
import json
import traceback

from .sac_agent import SACAgent
from .trading_environment import CryptoTradingEnvironment
from .reward_functions import RewardFunctions
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class MultiAgentTradingSystem:
    """
    Multi-agent system with comprehensive error handling.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize multi-agent system with validation."""
        # Device configuration
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing MultiAgentTradingSystem on device: {self.device}")
        
        # Specialized agents
        self.agents = {
            'trend_follower': None,
            'mean_reverter': None,
            'volatility_trader': None,
            'momentum_trader': None
        }
        
        # Meta-agent for allocation
        self.meta_agent = None
        self.agent_performance = {name: [] for name in self.agents.keys()}
        
        # Current environment reference
        self.current_env = None
        
        # Error tracking
        self.training_errors = 0
        self.max_training_errors = 50
        
    def train_agents(
        self, 
        market_data: pd.DataFrame, 
        training_episodes: int = 1000,
        save_interval: int = 100
    ) -> None:
        """Train agents with comprehensive error handling."""
        logger.info("Training multi-agent system with SAC agents...")
        
        # [ERROR-HANDLING] Validate inputs
        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            raise ValueError("Invalid market data for training")
        
        if training_episodes <= 0:
            raise ValueError(f"Invalid training episodes: {training_episodes}")
        
        for agent_name in self.agents.keys():
            try:
                logger.info(f"Training {agent_name} with SAC...")
                
                # Create environment with error handling
                env = CryptoTradingEnvironment(market_data)
                self.current_env = env
                
                # Bind appropriate reward function
                self._bind_reward_function(env, agent_name)
                
                # Create SAC agent
                agent = SACAgent(
                    state_size=env.state_size,
                    action_size=env.action_space,
                    continuous_action_dim=1,
                    hidden_dim=256,
                    device=self.device
                )
                
                # Training loop with error handling
                for episode in range(training_episodes):
                    try:
                        state = env.reset()
                        done = False
                        episode_reward = 0
                        steps = 0
                        max_steps = len(market_data) - 61  # Account for warmup
                        
                        while not done and steps < max_steps:
                            action = agent.act(state)
                            next_state, reward, done, info = env.step(action)
                            
                            # [ERROR-HANDLING] Validate step results
                            if not isinstance(reward, (int, float)) or not np.isfinite(reward):
                                logger.warning(f"Invalid reward: {reward}")
                                reward = -1.0
                            
                            agent.remember(state, action, reward, next_state, done)
                            
                            state = next_state
                            episode_reward += reward
                            steps += 1
                            
                            # Train periodically
                            if len(agent.memory) > 1000 and steps % 4 == 0:
                                agent.train(batch_size=256)
                        
                        # Track performance
                        self.agent_performance[agent_name].append(episode_reward)
                        
                        # Logging
                        if episode % 10 == 0:
                            self._log_training_progress(agent_name, episode, training_episodes, 
                                                      episode_reward, info, agent)
                        
                        # Save periodically
                        if episode % save_interval == 0 and episode > 0:
                            self._save_agent(agent, agent_name, episode)
                            
                    except Exception as e:
                        logger.error(f"Error in episode {episode} for {agent_name}: {e}")
                        self.training_errors += 1
                        
                        if self.training_errors > self.max_training_errors:
                            logger.error("Too many training errors, stopping")
                            raise
                
                self.agents[agent_name] = agent
                logger.info(f"Completed training {agent_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {agent_name}: {e}")
                logger.error(traceback.format_exc())
                # Continue with other agents
    
    def _bind_reward_function(self, env: CryptoTradingEnvironment, agent_name: str):
        """Bind appropriate reward function to environment"""
        self.current_env = env  # Store reference for reward functions
        
        if agent_name == 'trend_follower':
            env._calculate_sophisticated_reward = lambda es, tc: RewardFunctions.trend_following_reward(env, es, tc)
        elif agent_name == 'mean_reverter':
            env._calculate_sophisticated_reward = lambda es, tc: RewardFunctions.mean_reversion_reward(env, es, tc)
        elif agent_name == 'volatility_trader':
            env._calculate_sophisticated_reward = lambda es, tc: RewardFunctions.volatility_trading_reward(env, es, tc)
        elif agent_name == 'momentum_trader':
            env._calculate_sophisticated_reward = lambda es, tc: RewardFunctions.momentum_trading_reward(env, es, tc)
    
    def _log_training_progress(self, agent_name: str, episode: int, total_episodes: int,
                             episode_reward: float, info: Dict, agent: SACAgent):
        """Log training progress with error handling"""
        try:
            recent_rewards = self.agent_performance[agent_name][-10:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            logger.info(
                f"{agent_name} - Episode {episode}/{total_episodes}, "
                f"Reward: {episode_reward:.2f}, Avg: {avg_reward:.2f}, "
                f"Portfolio: ${info.get('portfolio_value', 0):.2f}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )
        except Exception as e:
            logger.error(f"Error logging progress: {e}")
    
    def _save_agent(self, agent: SACAgent, agent_name: str, episode: int):
        """Save agent with error handling"""
        try:
            save_dir = "agent_models"
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{agent_name}_sac_episode_{episode}.pt")
            agent.save(save_path)
            
        except Exception as e:
            logger.error(f"Error saving {agent_name}: {e}")
    
    def get_ensemble_action(self, state: np.ndarray, market_conditions: Dict) -> int:
        """Get ensemble action with comprehensive error handling."""
        try:
            # [ERROR-HANDLING] Validate inputs
            if not isinstance(state, np.ndarray):
                logger.warning("Invalid state type for ensemble action")
                return 0
            
            if not self._validate_agents_ready():
                logger.warning("Agents not ready for ensemble action")
                return 0
            
            agent_actions = {}
            agent_confidences = {}
            
            # Collect actions from each agent
            for agent_name, agent in self.agents.items():
                if agent is not None:
                    try:
                        action = agent.predict(state) if hasattr(agent, 'predict') else agent.act(state)
                        confidence = agent.get_confidence(state) if hasattr(agent, 'get_confidence') else 0.5
                        
                        # Validate action and confidence
                        if 0 <= action < 7:
                            agent_actions[agent_name] = action
                            agent_confidences[agent_name] = np.clip(confidence, 0, 1)
                            
                    except Exception as e:
                        logger.error(f"Error getting action from {agent_name}: {e}")
            
            if not agent_actions:
                logger.warning("No valid actions from agents")
                return 0
            
            # Calculate dynamic weights
            weights = self._calculate_agent_weights_safe(market_conditions)
            
            # Weighted voting
            action_scores = {}
            for agent_name, action in agent_actions.items():
                weight = weights.get(agent_name, 0)
                confidence = agent_confidences.get(agent_name, 0.5)
                
                adjusted_weight = weight * confidence
                
                if action not in action_scores:
                    action_scores[action] = 0
                action_scores[action] += adjusted_weight
            
            # Return action with highest score
            if action_scores:
                best_action = max(action_scores.items(), key=lambda x: x[1])[0]
                return best_action
            
            return 0
            
        except Exception as e:
            logger.error(f"Error in ensemble action: {e}")
            return 0
    
    def _validate_agents_ready(self) -> bool:
        """Check if agents are ready for predictions"""
        ready_count = sum(1 for agent in self.agents.values() if agent is not None)
        return ready_count >= 2  # At least 2 agents ready
    
    def _calculate_agent_weights_safe(self, market_conditions: Dict) -> Dict[str, float]:
        """Calculate agent weights with error handling"""
        try:
            # Base weights
            weights = {
                'trend_follower': 0.25,
                'mean_reverter': 0.25,
                'volatility_trader': 0.25,
                'momentum_trader': 0.25
            }
            
            # Get recent performance
            performance_scores = {}
            for agent_name, performance in self.agent_performance.items():
                if len(performance) >= 10:
                    recent_perf = performance[-50:]
                    avg_performance = np.mean(recent_perf)
                    performance_scores[agent_name] = max(0, avg_performance)
                else:
                    performance_scores[agent_name] = 0
            
            # Market condition adjustments
            volatility = market_conditions.get('volatility', 0.02)
            volatility = np.clip(volatility, 0, 0.2)  # Cap at 20%
            
            trend_strength = market_conditions.get('trend_strength', 0)
            trend_strength = np.clip(trend_strength, -1, 1)
            
            # Adjust based on conditions
            if volatility > 0.04:
                weights['volatility_trader'] *= 1.5
                weights['mean_reverter'] *= 0.7
            elif volatility < 0.015:
                weights['mean_reverter'] *= 1.4
                weights['volatility_trader'] *= 0.6
            
            if abs(trend_strength) > 0.02:
                weights['trend_follower'] *= 1.3
                weights['momentum_trader'] *= 1.2
                weights['mean_reverter'] *= 0.6
            
            # Performance adjustments
            total_performance = sum(performance_scores.values())
            if total_performance > 0:
                for agent_name in weights.keys():
                    performance_weight = performance_scores.get(agent_name, 0) / total_performance
                    weights[agent_name] = weights[agent_name] * 0.7 + performance_weight * 0.3
            
            # Normalize
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating agent weights: {e}")
            # Return equal weights as fallback
            return {name: 0.25 for name in self.agents.keys()}
    
    def save_all_agents(self, directory: str = "agent_models/"):
        """Save all agents with error handling"""
        try:
            os.makedirs(directory, exist_ok=True)
            
            saved_count = 0
            for agent_name, agent in self.agents.items():
                if agent is not None:
                    try:
                        filepath = os.path.join(directory, f"{agent_name}_final.pt")
                        agent.save(filepath)
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"Error saving {agent_name}: {e}")
            
            logger.info(f"Saved {saved_count}/{len(self.agents)} agents to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving agents: {e}")
    
    def load_all_agents(self, directory: str = "agent_models/"):
        """Load all agents with error handling"""
        try:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                return
            
            loaded_count = 0
            for agent_name in self.agents.keys():
                filepath = os.path.join(directory, f"{agent_name}_final.pt")
                
                if os.path.exists(filepath):
                    try:
                        # Create agent if not exists
                        if self.agents[agent_name] is None:
                            # Load metadata to get state size
                            metadata_path = filepath.replace('.pt', '_metadata.json')
                            if os.path.exists(metadata_path):
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                state_size = metadata.get('state_size', 75)
                                action_size = metadata.get('action_size', 7)
                            else:
                                state_size = 75
                                action_size = 7
                            
                            self.agents[agent_name] = SACAgent(
                                state_size=state_size,
                                action_size=action_size,
                                device=self.device
                            )
                        
                        self.agents[agent_name].load(filepath)
                        loaded_count += 1
                        logger.info(f"Loaded {agent_name} from {filepath}")
                        
                    except Exception as e:
                        logger.error(f"Error loading {agent_name}: {e}")
                else:
                    logger.warning(f"No saved model found for {agent_name}")
            
            logger.info(f"Loaded {loaded_count}/{len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error loading agents: {e}")