# Reinforcement Learning Module

## Overview
The reinforcement learning module implements a multi-agent system using Soft Actor-Critic (SAC) algorithms for adaptive cryptocurrency trading strategies.

## Structure
```
reinforcement_learning/
├── __init__.py                    # Package initialization
├── multi_agent_system.py          # Main multi-agent coordinator
├── risk_aware_extensions.py       # Risk-aware RL modifications
├── README.md                     # This file
└── modules/                      # Submodules
    ├── __init__.py
    ├── sac_agent.py             # SAC agent implementation
    ├── trading_environment.py    # Custom trading environment
    ├── multi_agent_coordinator.py # Agent coordination
    ├── reward_functions.py       # Reward calculations
    ├── replay_buffer.py         # Experience replay
    ├── network_definitions.py    # Neural networks
    └── data_classes.py          # Data structures
```

## Components

### MultiAgentTradingSystem
Coordinates multiple specialized RL agents:
- **Long-term Agent**: Strategic position management
- **Short-term Agent**: Tactical trading decisions
- **Risk Agent**: Dynamic risk adjustment
- **Market Making Agent**: Liquidity provision

### SACAgent
Soft Actor-Critic implementation with:
- Automatic temperature tuning
- Double Q-learning for stability
- Continuous action spaces
- Experience replay buffer

### TradingEnvironment
Custom Gym-compatible environment:
- Multi-asset trading support
- Realistic transaction costs
- Market impact modeling
- Portfolio constraints

### RewardFunctions
Various reward formulations:
- Sharpe ratio optimization
- Risk-adjusted returns
- Drawdown penalties
- Transaction cost awareness

## Key Features

### 1. Multi-Agent Architecture
```python
agents = {
    'long_term': SACAgent(state_dim=100, action_dim=4),
    'short_term': SACAgent(state_dim=80, action_dim=4),
    'risk': SACAgent(state_dim=60, action_dim=2),
    'market_making': SACAgent(state_dim=120, action_dim=6)
}
```

### 2. Risk-Aware Extensions
- Value-at-Risk (VaR) constraints
- Maximum drawdown limits
- Position concentration rules
- Correlation-based diversification

### 3. Advanced Reward Shaping
- Multi-objective optimization
- Curriculum learning support
- Adaptive reward scaling
- Exploration bonuses

## Usage

```python
from models.reinforcement_learning import MultiAgentTradingSystem

# Initialize system
rl_system = MultiAgentTradingSystem(
    device='cuda',
    learning_rate=3e-4,
    batch_size=256
)

# Get trading action
state = get_current_state()
action = await rl_system.get_action(state)

# Update with experience
reward = calculate_reward()
next_state = get_next_state()
rl_system.update(state, action, reward, next_state)
```

## Training

### Online Learning
The system supports continuous learning from live trading:
```python
# During trading loop
experience = Experience(state, action, reward, next_state, done)
rl_system.add_experience(experience)

# Periodic updates
if step % update_frequency == 0:
    rl_system.train_step()
```

### Offline Training
Pre-train on historical data:
```python
# Load historical data
dataset = load_historical_trades()

# Train agents
rl_system.train_offline(dataset, epochs=100)
```

## Configuration

Key parameters in config.yaml:
```yaml
reinforcement_learning:
  learning_rate: 3e-4
  batch_size: 256
  replay_buffer_size: 1000000
  gamma: 0.99
  tau: 0.005
  alpha: 0.2
  automatic_entropy_tuning: true
  reward_scale: 5.0
  hidden_sizes: [256, 256]
```

## Performance Optimization

### GPU Acceleration
- Automatic GPU detection
- Batch processing for efficiency
- Mixed precision training option

### Memory Management
- Circular replay buffer
- Gradient accumulation
- Model checkpointing

## Monitoring

Track agent performance with:
- Average reward trends
- Policy entropy
- Q-value estimates
- Action distributions
- Learning curves

## Model Persistence

Save and load trained agents:
```python
# Save
rl_system.save_all_agents("models/rl_agents/")

# Load
rl_system.load_all_agents("models/rl_agents/")
```

## Research Extensions

The module supports research into:
- Multi-agent coordination strategies
- Novel reward formulations
- Environment dynamics modeling
- Transfer learning approaches
- Meta-learning algorithms

## Dependencies

- PyTorch for neural networks
- Gym for environment interface
- NumPy for numerical operations
- Custom trading modules

## Testing

Run comprehensive tests:
```bash
pytest tests/models/test_reinforcement_learning.py
```

## Contributing

When extending the RL module:
1. Follow the SAC algorithm specifications
2. Maintain environment compatibility
3. Document reward functions clearly
4. Add performance benchmarks
5. Include ablation studies
