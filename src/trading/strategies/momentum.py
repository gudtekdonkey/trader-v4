import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import talib
from ..risk_manager import RiskManager
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class MomentumSignal:
    symbol: str
    direction: int  # 1 for long, -1 for short
    strength: float  # 0 to 1
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timeframe: str
    indicators: Dict[str, float]

class MomentumStrategy:
    """Advanced momentum trading strategy"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.active_positions = {}
        
        # Strategy parameters
        self.params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'adx_threshold': 25,
            'volume_multiplier': 1.5,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'min_confidence': 0.6
        }
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
    def analyze(self, df: pd.DataFrame, ml_predictions: Optional[Dict] = None) -> List[MomentumSignal]:
        """Analyze market data for momentum signals"""
        signals = []
        
        # Calculate indicators
        indicators = self._calculate_indicators(df)
        
        # Get current values
        current_idx = -1
        current_price = df['close'].iloc[current_idx]
        
        # Check for momentum signals
        if self._check_bullish_momentum(indicators, current_idx):
            signal = self._create_signal(
                df, indicators, current_idx, direction=1, ml_predictions=ml_predictions
            )
            if signal and signal.confidence >= self.params['min_confidence']:
                signals.append(signal)
                
        elif self._check_bearish_momentum(indicators, current_idx):
            signal = self._create_signal(
                df, indicators, current_idx, direction=-1, ml_predictions=ml_predictions
            )
            if signal and signal.confidence >= self.params['min_confidence']:
                signals.append(signal)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum indicators"""
        indicators = {}
        
        # Price momentum
        indicators['rsi'] = talib.RSI(df['close'], timeperiod=self.params['rsi_period'])
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=self.params['macd_fast'],
            slowperiod=self.params['macd_slow'],
            signalperiod=self.params['macd_signal']
        )
        
        # Trend strength
        indicators['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.params['adx_period'])
        indicators['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=self.params['adx_period'])
        indicators['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=self.params['adx_period'])
        
        # Volume confirmation
        indicators['volume_sma'] = df['volume'].rolling(20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        
        # Volatility
        indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Price action
        indicators['close_position'] = (df['close'] - df['low'].rolling(20).min()) / \
                                      (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        
        # Rate of change
        indicators['roc_5'] = talib.ROC(df['close'], timeperiod=5)
        indicators['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        indicators['roc_20'] = talib.ROC(df['close'], timeperiod=20)
        
        # Moving averages
        indicators['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        indicators['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        indicators['ema_9'] = talib.EMA(df['close'], timeperiod=9)
        indicators['ema_21'] = talib.EMA(df['close'], timeperiod=21)
        
        return indicators
    
    def _check_bullish_momentum(self, indicators: Dict[str, pd.Series], idx: int) -> bool:
        """Check for bullish momentum conditions"""
        # Strong trend
        if indicators['adx'].iloc[idx] < self.params['adx_threshold']:
            return False
        
        # Bullish directional movement
        if indicators['plus_di'].iloc[idx] <= indicators['minus_di'].iloc[idx]:
            return False
        
        # MACD bullish crossover or positive histogram
        macd_bullish = (indicators['macd'].iloc[idx] > indicators['macd_signal'].iloc[idx] and
                       indicators['macd_hist'].iloc[idx] > 0 and
                       indicators['macd_hist'].iloc[idx] > indicators['macd_hist'].iloc[idx-1])
        
        # RSI not overbought
        rsi_ok = indicators['rsi'].iloc[idx] < self.params['rsi_overbought']
        
        # Volume confirmation
        volume_confirm = indicators['volume_ratio'].iloc[idx] > self.params['volume_multiplier']
        
        # Price above moving averages
        price_above_ma = (indicators['ema_9'].iloc[idx] > indicators['ema_21'].iloc[idx] and
                         indicators['sma_20'].iloc[idx] > indicators['sma_50'].iloc[idx])
        
        # Positive rate of change
        positive_roc = (indicators['roc_5'].iloc[idx] > 0 and
                       indicators['roc_10'].iloc[idx] > 0)
        
        return macd_bullish and rsi_ok and volume_confirm and price_above_ma and positive_roc
    
    def _check_bearish_momentum(self, indicators: Dict[str, pd.Series], idx: int) -> bool:
        """Check for bearish momentum conditions"""
        # Strong trend
        if indicators['adx'].iloc[idx] < self.params['adx_threshold']:
            return False
        
        # Bearish directional movement
        if indicators['minus_di'].iloc[idx] <= indicators['plus_di'].iloc[idx]:
            return False
        
        # MACD bearish crossover or negative histogram
        macd_bearish = (indicators['macd'].iloc[idx] < indicators['macd_signal'].iloc[idx] and
                       indicators['macd_hist'].iloc[idx] < 0 and
                       indicators['macd_hist'].iloc[idx] < indicators['macd_hist'].iloc[idx-1])
        
        # RSI not oversold
        rsi_ok = indicators['rsi'].iloc[idx] > self.params['rsi_oversold']
        
        # Volume confirmation
        volume_confirm = indicators['volume_ratio'].iloc[idx] > self.params['volume_multiplier']
        
        # Price below moving averages
        price_below_ma = (indicators['ema_9'].iloc[idx] < indicators['ema_21'].iloc[idx] and
                         indicators['sma_20'].iloc[idx] < indicators['sma_50'].iloc[idx])
        
        # Negative rate of change
        negative_roc = (indicators['roc_5'].iloc[idx] < 0 and
                       indicators['roc_10'].iloc[idx] < 0)
        
        return macd_bearish and rsi_ok and volume_confirm and price_below_ma and negative_roc
    
    def _create_signal(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], 
                      idx: int, direction: int, ml_predictions: Optional[Dict] = None) -> Optional[MomentumSignal]:
        """Create a momentum signal"""
        current_price = df['close'].iloc[idx]
        atr = indicators['atr'].iloc[idx]
        
        # Calculate stop loss and take profit
        if direction == 1:  # Long
            stop_loss = current_price - (atr * self.params['atr_multiplier_sl'])
            take_profit = current_price + (atr * self.params['atr_multiplier_tp'])
        else:  # Short
            stop_loss = current_price + (atr * self.params['atr_multiplier_sl'])
            take_profit = current_price - (atr * self.params['atr_multiplier_tp'])
        
        # Calculate signal strength
        strength = self._calculate_signal_strength(indicators, idx, direction)
        
        # Calculate confidence
        confidence = strength
        if ml_predictions:
            ml_confidence = ml_predictions.get('confidence', 0.5)
            ml_direction = ml_predictions.get('direction', 0)
            
            # Boost confidence if ML agrees
            if ml_direction == direction:
                confidence = (confidence + ml_confidence) / 2
            else:
                confidence *= 0.7  # Reduce confidence if ML disagrees
        
        # Check risk limits
        position_size = self.risk_manager.calculate_position_size(
            current_price, stop_loss, df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTC'
        )
        
        if position_size <= 0:
            return None
        
        return MomentumSignal(
            symbol=df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTC',
            direction=direction,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timeframe='1h',
            indicators={
                'rsi': indicators['rsi'].iloc[idx],
                'macd': indicators['macd'].iloc[idx],
                'adx': indicators['adx'].iloc[idx],
                'volume_ratio': indicators['volume_ratio'].iloc[idx]
            }
        )
    
    def _calculate_signal_strength(self, indicators: Dict[str, pd.Series], idx: int, direction: int) -> float:
        """Calculate momentum signal strength"""
        strength_components = []
        
        # ADX strength (0-100 scaled to 0-1)
        adx_strength = min(indicators['adx'].iloc[idx] / 50, 1.0)
        strength_components.append(adx_strength)
        
        # RSI strength
        if direction == 1:  # Long
            rsi_strength = max(0, (indicators['rsi'].iloc[idx] - 30) / 40)  # 30-70 range
        else:  # Short
            rsi_strength = max(0, (70 - indicators['rsi'].iloc[idx]) / 40)  # 70-30 range
        strength_components.append(rsi_strength)
        
        # MACD histogram strength
        macd_hist_strength = min(abs(indicators['macd_hist'].iloc[idx]) / 
                                (abs(indicators['macd'].iloc[idx]) + 1e-10), 1.0)
        strength_components.append(macd_hist_strength)
        
        # Volume strength
        volume_strength = min(indicators['volume_ratio'].iloc[idx] / 3, 1.0)
        strength_components.append(volume_strength)
        
        # Rate of change strength
        roc_strength = min(abs(indicators['roc_10'].iloc[idx]) / 10, 1.0)
        strength_components.append(roc_strength)
        
        # Average all components
        return np.mean(strength_components)
    
    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Update existing positions and check for exits"""
        actions = []
        
        for symbol, position in list(self.active_positions.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Calculate unrealized PnL
                if position['direction'] == 1:  # Long
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:  # Short
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # Trailing stop loss
                if pnl_pct > 0.02:  # 2% profit
                    new_stop = self._calculate_trailing_stop(
                        position, current_price, pnl_pct
                    )
                    if new_stop != position['stop_loss']:
                        actions.append({
                            'action': 'update_stop_loss',
                            'symbol': symbol,
                            'new_stop': new_stop,
                            'old_stop': position['stop_loss']
                        })
                        position['stop_loss'] = new_stop
                
                # Check for manual exit conditions
                if self._should_exit_position(position, current_price, pnl_pct):
                    actions.append({
                        'action': 'close_position',
                        'symbol': symbol,
                        'reason': 'momentum_exhausted'
                    })
        
        return actions
    
    def _calculate_trailing_stop(self, position: Dict, current_price: float, pnl_pct: float) -> float:
        """Calculate trailing stop loss"""
        if position['direction'] == 1:  # Long
            # Move stop to breakeven at 2% profit
            if pnl_pct >= 0.02 and position['stop_loss'] < position['entry_price']:
                return position['entry_price'] * 1.001  # Small buffer above entry
            
            # Trail stop at 50% of profit
            if pnl_pct >= 0.05:
                trail_distance = (current_price - position['entry_price']) * 0.5
                new_stop = position['entry_price'] + trail_distance
                return max(new_stop, position['stop_loss'])
        
        else:  # Short
            # Move stop to breakeven at 2% profit
            if pnl_pct >= 0.02 and position['stop_loss'] > position['entry_price']:
                return position['entry_price'] * 0.999  # Small buffer below entry
            
            # Trail stop at 50% of profit
            if pnl_pct >= 0.05:
                trail_distance = (position['entry_price'] - current_price) * 0.5
                new_stop = position['entry_price'] - trail_distance
                return min(new_stop, position['stop_loss'])
        
        return position['stop_loss']
    
    def _should_exit_position(self, position: Dict, current_price: float, pnl_pct: float) -> bool:
        """Check if position should be exited based on momentum exhaustion"""
        # Time-based exit (momentum typically exhausts after certain period)
        time_in_position = pd.Timestamp.now() - position['entry_time']
        if time_in_position > pd.Timedelta(hours=24):
            return True
        
        # Momentum reversal exit (would need current indicators)
        # This is simplified - in practice you'd recalculate indicators
        
        return False
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return {
            'name': 'Momentum',
            'performance': self.performance,
            'active_positions': len(self.active_positions),
            'parameters': self.params
        }
