"""
ML prediction module for the trading bot.
Handles ML model predictions including ensemble and RL models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger(__name__)


class MLPredictor:
    """Manages ML predictions for trading decisions"""
    
    def __init__(self, ensemble_predictor, rl_system, regime_detector):
        self.ensemble_predictor = ensemble_predictor
        self.rl_system = rl_system
        self.regime_detector = regime_detector
        
    async def get_predictions(self, market_data: Dict) -> Dict:
        """Get ML model predictions including RL agents"""
        try:
            # Get ensemble predictions
            ensemble_predictions = await self._get_ensemble_predictions(market_data)
            
            # Get RL agent predictions
            rl_predictions = await self._get_rl_predictions(market_data)
            
            # Get regime predictions
            regime = self._get_regime_prediction(market_data)
            
            # Combine all predictions
            current_price = market_data['current']['mid_price']
            pred_price = ensemble_predictions.get('predicted_price', current_price)
            
            predictions = {
                'price_prediction': pred_price,
                'price_change': (pred_price - current_price) / current_price if current_price > 0 else 0,
                'confidence': ensemble_predictions.get('confidence', 0),
                'direction': 1 if pred_price > current_price else -1,
                'upper_bound': ensemble_predictions.get('upper_bound', pred_price * 1.02),
                'lower_bound': ensemble_predictions.get('lower_bound', pred_price * 0.98),
                'rl_action': rl_predictions.get('action', 0),
                'rl_confidence': rl_predictions.get('confidence', 0),
                'rl_agent_weights': rl_predictions.get('agent_weights', {}),
                'regime': regime
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting ML predictions: {e}")
            return self._get_default_predictions(market_data)
    
    async def _get_ensemble_predictions(self, market_data: Dict) -> Dict:
        """Get predictions from ensemble models"""
        try:
            df = market_data['ohlcv']
            current_data = market_data['current']
            
            # Prepare features for prediction
            features = self._prepare_ensemble_features(df, current_data)
            
            # Get predictions from ensemble
            predictions = self.ensemble_predictor.predict(features)
            
            # Calculate confidence based on prediction variance
            confidence = 1 - (predictions.get('std', 0) / predictions.get('mean', 1))
            
            return {
                'predicted_price': predictions.get('mean', current_data['mid_price']),
                'upper_bound': predictions.get('upper_95', current_data['mid_price'] * 1.02),
                'lower_bound': predictions.get('lower_95', current_data['mid_price'] * 0.98),
                'std': predictions.get('std', 0),
                'confidence': max(0, min(1, confidence))
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble predictions: {e}")
            return {}
    
    async def _get_rl_predictions(self, market_data: Dict) -> Dict:
        """Get predictions from multi-agent RL system"""
        try:
            df = market_data['ohlcv']
            current_data = market_data['current']
            
            # Prepare state for RL agents
            state = self._prepare_rl_state(df, current_data)
            
            # Get market conditions for agent selection
            market_conditions = {
                'volatility': current_data.get('volatility', 0.02),
                'trend_strength': df.get('trend_strength', pd.Series([0])).iloc[-1] if 'trend_strength' in df else 0,
                'regime': self._get_regime_prediction(market_data)
            }
            
            # Get ensemble action from all RL agents
            action, metadata = self.rl_system.get_ensemble_action(
                state=state,
                market_conditions=market_conditions
            )
            
            # Get individual agent predictions for analysis
            agent_actions = {}
            agent_confidences = {}
            
            for agent_name, agent in self.rl_system.agents.items():
                if agent is not None:
                    try:
                        agent_action = agent.predict(state)
                        agent_confidence = agent.get_confidence(state)
                        agent_actions[agent_name] = agent_action
                        agent_confidences[agent_name] = agent_confidence
                    except Exception as e:
                        logger.warning(f"Error getting prediction from {agent_name}: {e}")
            
            # Get agent weights
            agent_weights = self.rl_system._calculate_agent_weights(market_conditions)
            
            return {
                'action': action,
                'confidence': np.mean(list(agent_confidences.values())) if agent_confidences else 0,
                'agent_actions': agent_actions,
                'agent_confidences': agent_confidences,
                'agent_weights': agent_weights,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting RL predictions: {e}")
            return {'action': 0, 'confidence': 0}
    
    def _prepare_ensemble_features(self, df: pd.DataFrame, current_data: Dict) -> np.ndarray:
        """Prepare features for ensemble models"""
        try:
            # This should match your ensemble model's expected features
            features = []
            
            # Price features
            if len(df) >= 20:
                features.extend([
                    df['close'].pct_change(1).iloc[-1],
                    df['close'].pct_change(5).iloc[-1],
                    df['close'].pct_change(20).iloc[-1],
                ])
            else:
                features.extend([0, 0, 0])
            
            # Technical indicators
            features.extend([
                current_data.get('rsi_14', 50) / 100,
                current_data.get('macd', 0),
                current_data.get('bb_position', 0.5),
                current_data.get('atr', 0.02),
                current_data.get('volume_ratio', 1)
            ])
            
            # Market microstructure
            features.extend([
                current_data.get('order_imbalance', 0),
                current_data.get('spread', 0.001),
                current_data.get('depth_imbalance', 0)
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing ensemble features: {e}")
            return np.zeros((1, 20))  # Return default features
    
    def _prepare_rl_state(self, df: pd.DataFrame, current_data: Dict) -> np.ndarray:
        """Prepare state vector for RL agents (matching CryptoTradingEnvironment state)"""
        try:
            state_components = []
            
            # Price returns (last 15)
            if len(df) >= 20:
                price_returns = df['close'].pct_change().iloc[-15:].fillna(0).values
            else:
                price_returns = np.zeros(15)
            state_components.append(price_returns)
            
            # Technical indicators (5)
            tech_features = np.array([
                current_data.get('rsi_14', 50) / 100,
                current_data.get('macd', 0) / 100,
                current_data.get('bb_position', 0.5),
                current_data.get('atr', 0.02),
                current_data.get('volume_ratio', 1)
            ])
            state_components.append(tech_features)
            
            # Microstructure features (5)
            micro_features = np.array([
                current_data.get('order_imbalance', 0),
                current_data.get('spread', 0.001),
                current_data.get('pressure_ratio', 1),
                current_data.get('depth_imbalance', 0),
                current_data.get('estimated_price_impact', 0)
            ])
            state_components.append(micro_features)
            
            # Alternative data (4)
            alt_features = np.array([
                current_data.get('social_sentiment', 0),
                current_data.get('fear_greed_index', 50) / 100,
                current_data.get('whale_movements', 0),
                current_data.get('exchange_flows', 0)
            ])
            state_components.append(alt_features)
            
            # Multi-timeframe features (5)
            mtf_features = np.array([
                float(current_data.get('htf_trend', False)),
                float(current_data.get('mtf_trend', False)),
                float(current_data.get('stf_trend', False)),
                current_data.get('trend_agreement', 0),
                current_data.get('momentum_divergence', 0)
            ])
            state_components.append(mtf_features)
            
            # Portfolio features (11) - placeholder, should come from risk_manager
            portfolio_features = np.zeros(11)
            state_components.append(portfolio_features)
            
            # Combine all features
            state = np.concatenate(state_components)
            
            # Ensure state size is 75
            if len(state) < 75:
                state = np.pad(state, (0, 75 - len(state)), 'constant')
            else:
                state = state[:75]
            
            return state.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preparing RL state: {e}")
            return np.zeros(75, dtype=np.float32)
    
    def _get_regime_prediction(self, market_data: Dict) -> str:
        """Get market regime prediction"""
        try:
            df = market_data['ohlcv']
            if len(df) < 50:
                return 'unknown'
                
            regime = self.regime_detector.detect_regime(df)
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return 'unknown'
    
    def _get_default_predictions(self, market_data: Dict) -> Dict:
        """Get default predictions when ML models fail"""
        current_price = market_data.get('current', {}).get('mid_price', 0)
        
        return {
            'price_prediction': current_price,
            'price_change': 0,
            'confidence': 0,
            'direction': 0,
            'upper_bound': current_price * 1.02,
            'lower_bound': current_price * 0.98,
            'rl_action': 0,
            'rl_confidence': 0,
            'rl_agent_weights': {},
            'regime': 'unknown'
        }


class RLSignalGenerator:
    """Generates trading signals from RL actions"""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        
    def create_signal(self, symbol: str, action: int, confidence: float, 
                     current_data: Dict, metadata: Dict = None):
        """Create trading signal from RL action"""
        try:
            # Map RL actions to trading signals
            # Actions: 0=hold, 1=buy25%, 2=buy50%, 3=buy100%, 4=sell25%, 5=sell50%, 6=sell100%
            
            if action == 0:
                return None
            
            # Determine direction and size
            if action <= 3:  # Buy actions
                direction = 1
                size_pct = [0.25, 0.5, 1.0][action - 1]
            else:  # Sell actions
                direction = -1
                size_pct = [0.25, 0.5, 1.0][action - 4]
            
            current_price = current_data['mid_price']
            
            # Check if risk-adjusted
            if metadata and metadata.get('adjusted'):
                if metadata.get('adjustment_reason') not in ['position_size_reduced']:
                    logger.info(f"Skipping RL signal due to {metadata.get('adjustment_reason')}")
                    return None
            
            # Create signal object
            signal = RLSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                entry_price=current_price * (1 + 0.0001 * direction),  # Small slippage
                stop_loss=current_price * (1 - 0.02 * direction),  # 2% stop loss
                size_percentage=size_pct,
                metadata=metadata
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating RL signal: {e}")
            return None


class RLSignal:
    """RL-generated trading signal"""
    
    def __init__(self, symbol: str, direction: int, confidence: float,
                 entry_price: float, stop_loss: float, size_percentage: float,
                 metadata: Dict = None):
        self.symbol = symbol
        self.direction = direction
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.size_percentage = size_percentage
        self.type = 'rl_ensemble'
        self.strength = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
