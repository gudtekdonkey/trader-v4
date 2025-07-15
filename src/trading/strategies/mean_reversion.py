import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import talib
from scipy import stats
from ..risk_manager import RiskManager
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class MeanReversionSignal:
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    stop_loss: float
    take_profit: float
    z_score: float
    confidence: float
    timeframe: str
    reversion_target: float
    indicators: Dict[str, float]

class MeanReversionStrategy:
    """Statistical arbitrage and mean reversion strategy"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.active_positions = {}
        
        # Strategy parameters
        self.params = {
            'lookback_period': 20,
            'entry_z_score': 2.0,
            'exit_z_score': 0.5,
            'max_z_score': 3.0,
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'min_volatility': 0.01,
            'max_volatility': 0.10,
            'atr_multiplier_sl': 3.0,
            'position_hold_bars': 48,  # Maximum bars to hold position
            'min_confidence': 0.65
        }
        
        # Statistical tracking
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'win_rate': 0
        }
        
    def analyze(self, df: pd.DataFrame, ml_predictions: Optional[Dict] = None) -> List[MeanReversionSignal]:
        """Analyze market for mean reversion opportunities"""
        signals = []
        
        # Calculate mean reversion indicators
        indicators = self._calculate_indicators(df)
        
        # Check if market conditions are suitable
        if not self._check_market_conditions(df, indicators):
            return signals
        
        # Get current values
        current_idx = -1
        current_price = df['close'].iloc[current_idx]
        z_score = indicators['z_score'].iloc[current_idx]
        
        # Check for mean reversion signals
        if abs(z_score) >= self.params['entry_z_score'] and abs(z_score) <= self.params['max_z_score']:
            # Oversold condition (long signal)
            if z_score <= -self.params['entry_z_score']:
                signal = self._create_signal(
                    df, indicators, current_idx, direction=1, ml_predictions=ml_predictions
                )
                if signal and signal.confidence >= self.params['min_confidence']:
                    signals.append(signal)
            
            # Overbought condition (short signal)
            elif z_score >= self.params['entry_z_score']:
                signal = self._create_signal(
                    df, indicators, current_idx, direction=-1, ml_predictions=ml_predictions
                )
                if signal and signal.confidence >= self.params['min_confidence']:
                    signals.append(signal)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate mean reversion indicators"""
        indicators = {}
        
        # Price statistics
        indicators['sma'] = df['close'].rolling(self.params['lookback_period']).mean()
        indicators['std'] = df['close'].rolling(self.params['lookback_period']).std()
        indicators['z_score'] = (df['close'] - indicators['sma']) / indicators['std']
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=self.params['bb_period'],
            nbdevup=self.params['bb_std'],
            nbdevdn=self.params['bb_std']
        )
        indicators['bb_position'] = (df['close'] - indicators['bb_lower']) / \
                                   (indicators['bb_upper'] - indicators['bb_lower'])
        
        # RSI for extremes
        indicators['rsi'] = talib.RSI(df['close'], timeperiod=self.params['rsi_period'])
        
        # Volatility
        indicators['volatility'] = df['returns'].rolling(20).std()
        indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume analysis
        indicators['volume_sma'] = df['volume'].rolling(20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        
        # Mean reversion speed
        indicators['half_life'] = self._calculate_half_life(df['close'], window=60)
        
        # Cointegration score (if multiple assets)
        indicators['hurst_exponent'] = self._calculate_hurst_exponent(df['close'], window=100)
        
        # Price efficiency ratio
        change = (df['close'] - df['close'].shift(self.params['lookback_period'])).abs()
        path = df['close'].diff().abs().rolling(self.params['lookback_period']).sum()
        indicators['efficiency_ratio'] = change / (path + 1e-10)
        
        return indicators
    
    def _calculate_half_life(self, prices: pd.Series, window: int = 60) -> pd.Series:
        """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process"""
        half_lives = pd.Series(index=prices.index, dtype=float)
        
        for i in range(window, len(prices)):
            y = prices.iloc[i-window:i].values
            y_lag = prices.iloc[i-window-1:i-1].values
            
            # Run regression: y_t = alpha + beta * y_{t-1} + epsilon
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                # Half-life = -log(2) / log(beta[1])
                if 0 < beta[1] < 1:
                    half_lives.iloc[i] = -np.log(2) / np.log(beta[1])
                else:
                    half_lives.iloc[i] = np.nan
            except:
                half_lives.iloc[i] = np.nan
        
        return half_lives.fillna(method='ffill')
    
    def _calculate_hurst_exponent(self, prices: pd.Series, window: int = 100) -> pd.Series:
        """Calculate Hurst exponent to measure mean reversion tendency"""
        hurst_values = pd.Series(index=prices.index, dtype=float)
        
        for i in range(window, len(prices)):
            data = prices.iloc[i-window:i].values
            
            # Calculate R/S statistic
            mean_data = np.mean(data)
            deviations = data - mean_data
            cumulative_deviations = np.cumsum(deviations)
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(data, ddof=1)
            
            if S != 0:
                RS = R / S
                # Hurst exponent approximation
                hurst_values.iloc[i] = np.log(RS) / np.log(window)
            else:
                hurst_values.iloc[i] = 0.5
        
        return hurst_values.fillna(0.5)
    
    def _check_market_conditions(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> bool:
        """Check if market conditions are suitable for mean reversion"""
        current_idx = -1
        
        # Check volatility is in acceptable range
        current_vol = indicators['volatility'].iloc[current_idx]
        if current_vol < self.params['min_volatility'] or current_vol > self.params['max_volatility']:
            return False
        
        # Check for trending market (mean reversion works better in ranging markets)
        efficiency_ratio = indicators['efficiency_ratio'].iloc[current_idx]
        if efficiency_ratio > 0.7:  # Strong trend
            return False
        
        # Check Hurst exponent (< 0.5 indicates mean reversion)
        hurst = indicators['hurst_exponent'].iloc[current_idx]
        if hurst > 0.6:  # Trending behavior
            return False
        
        # Check half-life is reasonable
        half_life = indicators['half_life'].iloc[current_idx]
        if pd.isna(half_life) or half_life > 48:  # Too slow mean reversion
            return False
        
        return True
    
    def _create_signal(self, df: pd.DataFrame, indicators: Dict[str, pd.Series],
                      idx: int, direction: int, ml_predictions: Optional[Dict] = None) -> Optional[MeanReversionSignal]:
        """Create mean reversion signal"""
        current_price = df['close'].iloc[idx]
        sma = indicators['sma'].iloc[idx]
        atr = indicators['atr'].iloc[idx]
        z_score = indicators['z_score'].iloc[idx]
        
        # Calculate reversion target
        reversion_target = sma
        
        # Calculate entry and exit levels
        if direction == 1:  # Long
            entry_price = current_price
            stop_loss = current_price - (atr * self.params['atr_multiplier_sl'])
            take_profit = reversion_target
        else:  # Short
            entry_price = current_price
            stop_loss = current_price + (atr * self.params['atr_multiplier_sl'])
            take_profit = reversion_target
        
        # Ensure favorable risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if reward < risk * 1.5:  # Minimum 1.5:1 R/R
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(indicators, idx, direction, ml_predictions)
        
        # Check position sizing
        position_size = self.risk_manager.calculate_position_size(
            entry_price, stop_loss, df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTC'
        )
        
        if position_size <= 0:
            return None
        
        return MeanReversionSignal(
            symbol=df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTC',
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            z_score=z_score,
            confidence=confidence,
            timeframe='1h',
            reversion_target=reversion_target,
            indicators={
                'z_score': z_score,
                'rsi': indicators['rsi'].iloc[idx],
                'bb_position': indicators['bb_position'].iloc[idx],
                'half_life': indicators['half_life'].iloc[idx],
                'hurst': indicators['hurst_exponent'].iloc[idx]
            }
        )
    
    def _calculate_confidence(self, indicators: Dict[str, pd.Series], idx: int, 
                            direction: int, ml_predictions: Optional[Dict] = None) -> float:
        """Calculate signal confidence"""
        confidence_components = []
        
        # Z-score confidence (stronger signal at higher z-scores)
        z_score_conf = min(abs(indicators['z_score'].iloc[idx]) / self.params['max_z_score'], 1.0)
        confidence_components.append(z_score_conf)
        
        # RSI confirmation
        rsi = indicators['rsi'].iloc[idx]
        if direction == 1 and rsi < self.params['rsi_oversold']:
            confidence_components.append(0.9)
        elif direction == -1 and rsi > self.params['rsi_overbought']:
            confidence_components.append(0.9)
        else:
            confidence_components.append(0.5)
        
        # Bollinger Band position
        bb_pos = indicators['bb_position'].iloc[idx]
        if (direction == 1 and bb_pos < 0.2) or (direction == -1 and bb_pos > 0.8):
            confidence_components.append(0.8)
        else:
            confidence_components.append(0.4)
        
        # Mean reversion speed (faster is better)
        half_life = indicators['half_life'].iloc[idx]
        if not pd.isna(half_life) and half_life < 24:
            confidence_components.append(0.8)
        else:
            confidence_components.append(0.4)
        
        # Hurst exponent (lower indicates mean reversion)
        hurst = indicators['hurst_exponent'].iloc[idx]
        if hurst < 0.4:
            confidence_components.append(0.9)
        elif hurst < 0.5:
            confidence_components.append(0.7)
        else:
            confidence_components.append(0.3)
        
        # Base confidence
        base_confidence = np.mean(confidence_components)
        
        # ML adjustment
        if ml_predictions:
            ml_conf = ml_predictions.get('confidence', 0.5)
            ml_direction = ml_predictions.get('direction', 0)
            
            if ml_direction == direction:
                final_confidence = (base_confidence + ml_conf) / 2
            else:
                final_confidence = base_confidence * 0.8
        else:
            final_confidence = base_confidence
        
        return final_confidence
    
    def update_positions(self, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Update positions based on current market data"""
        actions = []
        
        for symbol, position in list(self.active_positions.items()):
            if symbol in current_data:
                df = current_data[symbol]
                indicators = self._calculate_indicators(df)
                current_idx = -1
                
                current_price = df['close'].iloc[current_idx]
                current_z_score = indicators['z_score'].iloc[current_idx]
                
                # Update bars held
                position['bars_held'] += 1
                
                # Check exit conditions
                should_exit, exit_reason = self._check_exit_conditions(
                    position, current_price, current_z_score, indicators, current_idx
                )
                
                if should_exit:
                    actions.append({
                        'action': 'close_position',
                        'symbol': symbol,
                        'reason': exit_reason
                    })
                    
                    # Calculate and record PnL
                    if position['direction'] == 1:
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                    
                    self._update_stats(pnl)
        
        return actions
    
    def _check_exit_conditions(self, position: Dict, current_price: float, current_z_score: float,
                              indicators: Dict[str, pd.Series], idx: int) -> Tuple[bool, str]:
        """Check if position should be exited"""
        # Z-score crossed zero (mean reversion complete)
        if abs(current_z_score) <= self.params['exit_z_score']:
            return True, 'mean_reversion_complete'
        
        # Z-score reversed (beyond opposite threshold)
        if position['entry_z_score'] > 0 and current_z_score < -self.params['exit_z_score']:
            return True, 'z_score_reversal'
        elif position['entry_z_score'] < 0 and current_z_score > self.params['exit_z_score']:
            return True, 'z_score_reversal'
        
        # Maximum holding period reached
        if position['bars_held'] >= self.params['position_hold_bars']:
            return True, 'max_holding_period'
        
        # Volatility spike (market regime change)
        current_vol = indicators['volatility'].iloc[idx]
        if current_vol > self.params['max_volatility'] * 1.5:
            return True, 'volatility_spike'
        
        return False, ''
    
    def _update_stats(self, pnl: float):
        """Update strategy statistics"""
        if pnl > 0:
            self.stats['winning_trades'] += 1
            self.stats['avg_win'] = (self.stats['avg_win'] * (self.stats['winning_trades'] - 1) + pnl) / self.stats['winning_trades']
        else:
            losing_trades = self.stats['total_trades'] - self.stats['winning_trades']
            self.stats['avg_loss'] = (self.stats['avg_loss'] * (losing_trades - 1) + pnl) / losing_trades
        
        self.stats['total_pnl'] += pnl
        self.stats['win_rate'] = self.stats['winning_trades'] / self.stats['total_trades'] if self.stats['total_trades'] > 0 else 0
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return {
            'name': 'Mean Reversion',
            'stats': self.stats,
            'active_positions': len(self.active_positions),
            'parameters': self.params
        }
