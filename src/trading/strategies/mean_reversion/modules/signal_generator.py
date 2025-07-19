"""
Mean Reversion Strategy - Signal Generator Module
Handles signal generation and confidence calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import traceback
from ....utils.logger import setup_logger

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


class SignalGenerator:
    """Generate mean reversion trading signals"""
    
    def __init__(self, params: Dict, risk_manager):
        """
        Initialize signal generator
        
        Args:
            params: Strategy parameters
            risk_manager: Risk management instance
        """
        self.params = params
        self.risk_manager = risk_manager
        self.error_stats = {
            'signal_generation_errors': 0,
            'last_error': None
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], 
                        ml_predictions: Optional[Dict] = None) -> List[MeanReversionSignal]:
        """Generate trading signals based on indicators"""
        signals = []
        
        try:
            # Check if market conditions are suitable
            if not self._check_market_conditions(indicators):
                logger.debug("Market conditions not suitable for mean reversion")
                return signals
            
            # Get current values with validation
            current_idx = -1
            
            # Validate current price
            try:
                current_price = float(df['close'].iloc[current_idx])
                if not np.isfinite(current_price) or current_price <= 0:
                    raise ValueError(f"Invalid current price: {current_price}")
            except Exception as e:
                logger.error(f"Error getting current price: {e}")
                return signals
            
            # Get z-score safely
            try:
                z_score = float(indicators['z_score'].iloc[current_idx])
                if not np.isfinite(z_score):
                    logger.warning(f"Invalid z-score: {z_score}")
                    return signals
            except Exception as e:
                logger.error(f"Error getting z-score: {e}")
                return signals
            
            # Check for mean reversion signals
            if abs(z_score) >= self.params['entry_z_score'] and abs(z_score) <= self.params['max_z_score']:
                # Oversold condition (long signal)
                if z_score <= -self.params['entry_z_score']:
                    signal = self._create_signal(
                        df, indicators, current_idx, direction=1, ml_predictions=ml_predictions
                    )
                    if signal and signal.confidence >= self.params['min_confidence']:
                        signals.append(signal)
                        logger.info(f"Generated long mean reversion signal for {signal.symbol}, z-score: {z_score:.2f}")
                
                # Overbought condition (short signal)
                elif z_score >= self.params['entry_z_score']:
                    signal = self._create_signal(
                        df, indicators, current_idx, direction=-1, ml_predictions=ml_predictions
                    )
                    if signal and signal.confidence >= self.params['min_confidence']:
                        signals.append(signal)
                        logger.info(f"Generated short mean reversion signal for {signal.symbol}, z-score: {z_score:.2f}")
                        
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {e}")
            self.error_stats['signal_generation_errors'] += 1
            self.error_stats['last_error'] = str(e)
        
        return signals
    
    def _check_market_conditions(self, indicators: Dict[str, pd.Series]) -> bool:
        """Check if market conditions are suitable for mean reversion"""
        try:
            current_idx = -1
            
            # Validate indicators exist
            required_indicators = ['volatility', 'efficiency_ratio', 'hurst_exponent', 'half_life']
            for indicator in required_indicators:
                if indicator not in indicators or indicators[indicator] is None:
                    logger.warning(f"Missing indicator for market condition check: {indicator}")
                    return False
            
            # Check volatility is in acceptable range
            current_vol = float(indicators['volatility'].iloc[current_idx])
            if not np.isfinite(current_vol):
                logger.warning("Invalid volatility value")
                return False
                
            if current_vol < self.params['min_volatility'] or current_vol > self.params['max_volatility']:
                logger.debug(f"Volatility {current_vol:.4f} outside acceptable range")
                return False
            
            # Check for trending market (mean reversion works better in ranging markets)
            efficiency_ratio = float(indicators['efficiency_ratio'].iloc[current_idx])
            if efficiency_ratio > 0.7:  # Strong trend
                logger.debug(f"Market too trendy for mean reversion: efficiency ratio {efficiency_ratio:.2f}")
                return False
            
            # Check Hurst exponent (< 0.5 indicates mean reversion)
            hurst = float(indicators['hurst_exponent'].iloc[current_idx])
            if hurst > 0.6:  # Trending behavior
                logger.debug(f"Hurst exponent {hurst:.2f} indicates trending, not mean reverting")
                return False
            
            # Check half-life is reasonable
            half_life = float(indicators['half_life'].iloc[current_idx])
            if pd.isna(half_life) or half_life > 48:  # Too slow mean reversion
                logger.debug(f"Half-life {half_life} too long for mean reversion")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return False
    
    def _create_signal(self, df: pd.DataFrame, indicators: Dict[str, pd.Series],
                      idx: int, direction: int, ml_predictions: Optional[Dict] = None) -> Optional[MeanReversionSignal]:
        """Create mean reversion signal with comprehensive error handling"""
        try:
            # Validate inputs
            if idx < -len(df) or idx >= len(df):
                logger.error(f"Invalid index {idx} for DataFrame of length {len(df)}")
                return None
            
            current_price = float(df['close'].iloc[idx])
            sma = float(indicators['sma'].iloc[idx])
            atr = float(indicators['atr'].iloc[idx])
            z_score = float(indicators['z_score'].iloc[idx])
            
            # Validate indicator values
            if not all(np.isfinite(v) for v in [current_price, sma, atr, z_score]):
                logger.warning("Invalid indicator values for signal creation")
                return None
            
            if current_price <= 0 or atr <= 0:
                logger.warning("Invalid price or ATR for signal creation")
                return None
            
            # Calculate reversion target
            reversion_target = sma
            
            # Calculate entry and exit levels
            if direction == 1:  # Long
                entry_price = current_price
                stop_loss = max(0, current_price - (atr * self.params['atr_multiplier_sl']))
                take_profit = reversion_target
            else:  # Short
                entry_price = current_price
                stop_loss = current_price + (atr * self.params['atr_multiplier_sl'])
                take_profit = reversion_target
            
            # Ensure favorable risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk <= 0:
                logger.warning("Invalid risk calculation")
                return None
                
            if reward < risk * 1.5:  # Minimum 1.5:1 R/R
                logger.debug(f"Insufficient risk/reward ratio: {reward/risk:.2f}")
                return None
            
            # Calculate confidence
            confidence = self._calculate_confidence(indicators, idx, direction, ml_predictions)
            
            # Validate confidence
            if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
                logger.warning(f"Invalid confidence value: {confidence}")
                confidence = 0.5
            
            # Check position sizing
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTC'
            
            try:
                position_size = self.risk_manager.calculate_position_size(
                    entry_price, stop_loss, symbol
                )
                
                if position_size <= 0:
                    logger.warning("Position size calculation returned zero or negative")
                    return None
            except Exception as e:
                logger.error(f"Error calculating position size: {e}")
                return None
            
            # Validate indicator values for signal
            indicator_values = {}
            for key in ['z_score', 'rsi', 'bb_position', 'half_life', 'hurst_exponent']:
                try:
                    if key in indicators and indicators[key] is not None:
                        value = float(indicators[key].iloc[idx])
                        if np.isfinite(value):
                            indicator_values[key] = value
                        else:
                            indicator_values[key] = 0
                    else:
                        indicator_values[key] = 0
                except Exception:
                    indicator_values[key] = 0
            
            return MeanReversionSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                z_score=z_score,
                confidence=confidence,
                timeframe='1h',
                reversion_target=reversion_target,
                indicators=indicator_values
            )
            
        except Exception as e:
            logger.error(f"Error creating mean reversion signal: {e}")
            logger.error(traceback.format_exc())
            self.error_stats['signal_generation_errors'] += 1
            return None
    
    def _calculate_confidence(self, indicators: Dict[str, pd.Series], idx: int, 
                            direction: int, ml_predictions: Optional[Dict] = None) -> float:
        """Calculate signal confidence with error handling"""
        try:
            confidence_components = []
            
            # Z-score confidence (stronger signal at higher z-scores)
            try:
                z_score = abs(float(indicators['z_score'].iloc[idx]))
                z_score_conf = min(z_score / self.params['max_z_score'], 1.0)
                confidence_components.append(z_score_conf)
            except Exception:
                confidence_components.append(0.5)
            
            # RSI confirmation
            try:
                rsi = float(indicators['rsi'].iloc[idx])
                if direction == 1 and rsi < self.params['rsi_oversold']:
                    confidence_components.append(0.9)
                elif direction == -1 and rsi > self.params['rsi_overbought']:
                    confidence_components.append(0.9)
                else:
                    confidence_components.append(0.5)
            except Exception:
                confidence_components.append(0.5)
            
            # Bollinger Band position
            try:
                bb_pos = float(indicators['bb_position'].iloc[idx])
                if (direction == 1 and bb_pos < 0.2) or (direction == -1 and bb_pos > 0.8):
                    confidence_components.append(0.8)
                else:
                    confidence_components.append(0.4)
            except Exception:
                confidence_components.append(0.5)
            
            # Mean reversion speed (faster is better)
            try:
                half_life = float(indicators['half_life'].iloc[idx])
                if not pd.isna(half_life) and half_life < 24:
                    confidence_components.append(0.8)
                else:
                    confidence_components.append(0.4)
            except Exception:
                confidence_components.append(0.5)
            
            # Hurst exponent (lower indicates mean reversion)
            try:
                hurst = float(indicators['hurst_exponent'].iloc[idx])
                if hurst < 0.4:
                    confidence_components.append(0.9)
                elif hurst < 0.5:
                    confidence_components.append(0.7)
                else:
                    confidence_components.append(0.3)
            except Exception:
                confidence_components.append(0.5)
            
            # Base confidence
            if confidence_components:
                base_confidence = np.mean(confidence_components)
            else:
                base_confidence = 0.5
            
            # ML adjustment
            if ml_predictions:
                try:
                    ml_conf = float(ml_predictions.get('confidence', 0.5))
                    ml_direction = int(ml_predictions.get('direction', 0))
                    
                    if ml_direction == direction:
                        final_confidence = (base_confidence + ml_conf) / 2
                    else:
                        final_confidence = base_confidence * 0.8
                except Exception:
                    final_confidence = base_confidence
            else:
                final_confidence = base_confidence
            
            # Ensure valid confidence
            final_confidence = np.clip(final_confidence, 0, 1)
            
            return float(final_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
