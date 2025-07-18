"""
Risk-aware extensions for the MultiAgentTradingSystem
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskAwareTradingEnvironment:
    """
    Extension to make the trading environment risk-aware.
    Wraps the base environment and modifies actions based on risk constraints.
    """
    
    def __init__(self, base_env, risk_manager=None):
        """
        Initialize risk-aware wrapper.
        
        Args:
            base_env: Base CryptoTradingEnvironment instance
            risk_manager: Risk manager instance to check constraints
        """
        self.base_env = base_env
        self.risk_manager = risk_manager
        
        # Pass through attributes
        self.state_size = base_env.state_size
        self.action_space = base_env.action_space
        self.market_data = base_env.market_data
        
    def reset(self):
        """Reset the environment"""
        return self.base_env.reset()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action with risk constraints.
        
        Args:
            action: Integer action (0-6)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # If no risk manager, pass through
        if self.risk_manager is None:
            return self.base_env.step(action)
        
        # Get current state info
        current_price = self.base_env.market_data['close'].iloc[self.base_env.current_step]
        symbol = self.base_env.market_data.attrs.get('symbol', 'UNKNOWN')
        
        # Convert action to trade size
        trade_size = self.base_env._calculate_trade_size(action, current_price)
        
        if trade_size != 0:
            # Check with risk manager
            can_trade = self.risk_manager.check_trade(
                symbol=symbol,
                quantity=abs(trade_size),
                price=current_price,
                side='buy' if trade_size > 0 else 'sell'
            )
            
            if not can_trade:
                # Convert to hold action
                action = 0
                logger.info(f"Risk manager blocked trade action {action} for {symbol}")
                
                # Apply penalty for risk violation
                next_state, reward, done, info = self.base_env.step(0)
                reward -= 5  # Penalty for attempting risky trade
                info['risk_blocked'] = True
                return next_state, reward, done, info
        
        # Execute the action
        return self.base_env.step(action)
    
    def get_portfolio_value(self):
        """Get current portfolio value"""
        return self.base_env.get_portfolio_value()


def get_risk_adjusted_action(
    rl_system,
    state: np.ndarray,
    market_conditions: Dict,
    risk_manager,
    current_price: float,
    symbol: str
) -> Tuple[int, Dict[str, Any]]:
    """
    Get risk-adjusted action from RL system.
    
    Args:
        rl_system: MultiAgentTradingSystem instance
        state: Current state vector
        market_conditions: Market conditions dict
        risk_manager: Risk manager instance
        current_price: Current market price
        symbol: Trading symbol
        
    Returns:
        Tuple of (adjusted_action, metadata)
    """
    try:
        # Get ensemble action from RL system
        action = rl_system.get_ensemble_action(state, market_conditions)
        
        # Create metadata
        metadata = {
            'original_action': action,
            'adjusted': False,
            'adjustment_reason': None
        }
        
        # Map action to trade parameters
        action_map = {
            0: {'side': 'hold', 'size_pct': 0},
            1: {'side': 'buy', 'size_pct': 0.25},
            2: {'side': 'buy', 'size_pct': 0.50},
            3: {'side': 'buy', 'size_pct': 1.00},
            4: {'side': 'sell', 'size_pct': 0.25},
            5: {'side': 'sell', 'size_pct': 0.50},
            6: {'side': 'sell', 'size_pct': 1.00}
        }
        
        trade_params = action_map.get(action, {'side': 'hold', 'size_pct': 0})
        
        if trade_params['side'] != 'hold':
            # Calculate position size based on available capital
            portfolio_value = risk_manager.get_portfolio_value()
            available_capital = risk_manager.get_available_capital()
            
            if trade_params['side'] == 'buy':
                # For buy, use available capital
                trade_value = available_capital * trade_params['size_pct']
                quantity = trade_value / current_price
            else:
                # For sell, use current position
                current_position = risk_manager.positions.get(symbol, {}).get('quantity', 0)
                quantity = current_position * trade_params['size_pct']
            
            # Check with risk manager
            can_trade = risk_manager.check_trade(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                side=trade_params['side']
            )
            
            if not can_trade:
                # Check if position sizing would help
                max_quantity = risk_manager.get_max_position_size(symbol, current_price)
                
                if max_quantity > 0 and max_quantity < quantity:
                    # Reduce to maximum allowed
                    if trade_params['side'] == 'buy':
                        # Find appropriate reduced action
                        reduced_value = max_quantity * current_price
                        reduced_pct = reduced_value / available_capital
                        
                        if reduced_pct >= 0.75:
                            action = 3  # Buy 100% of allowed
                        elif reduced_pct >= 0.40:
                            action = 2  # Buy 50%
                        elif reduced_pct >= 0.15:
                            action = 1  # Buy 25%
                        else:
                            action = 0  # Too small, hold
                    else:
                        # For sell, find appropriate reduced action
                