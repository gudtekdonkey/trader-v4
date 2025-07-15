"""
Position Sizer Module

Advanced position sizing with multiple algorithms including Kelly Criterion,
volatility-based sizing, optimal f, and machine learning approaches.

Classes:
    PositionSizer: Main position sizing calculator with multiple methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PositionSizer:
    """
    Advanced position sizing with multiple algorithms.
    
    This class implements various position sizing methods including:
    - Fixed fractional sizing
    - Kelly Criterion
    - Volatility-based sizing
    - Optimal f
    - Risk parity
    - Machine learning-based sizing
    - Regime-adjusted sizing
    
    Attributes:
        risk_manager: Risk management instance
        params: Position sizing parameters
        sizing_history: Historical sizing decisions
    """
    
    def __init__(self, risk_manager) -> None:
        """
        Initialize the Position Sizer.
        
        Args:
            risk_manager: Risk management instance
        """
        self.risk_manager = risk_manager
        
        # Sizing parameters
        self.params: Dict[str, Any] = {
            'kelly_fraction': 0.25,  # Use 25% of Kelly for safety
            'max_position_pct': 0.1,  # Max 10% per position
            'min_position_size': 100,  # Minimum $100 position
            'confidence_threshold': 0.6,  # Minimum confidence for full size
            'volatility_lookback': 20,  # Days for volatility calculation
            'correlation_penalty': 0.3,  # Reduce size by 30% for correlated positions
            'regime_adjustments': {
                'low_volatility': 1.2,
                'normal': 1.0,
                'high_volatility': 0.6,
                'extreme_volatility': 0.3
            }
        }
        
        # Track sizing history
        self.sizing_history: List[Dict] = []
        
    def calculate_position_size(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, float], 
        portfolio_state: Dict[str, Any]
    ) -> float:
        """
        Calculate optimal position size using multiple methods.
        
        This method combines various sizing algorithms weighted by market
        conditions to determine the optimal position size.
        
        Args:
            signal: Trading signal with entry price, stop loss, confidence
            market_data: Current market conditions
            portfolio_state: Current portfolio positions and metrics
            
        Returns:
            Final position size in base currency
        """
        # Get base inputs
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss', entry_price * 0.98)
        confidence = signal.get('confidence', 0.5)
        symbol = signal['symbol']
        
        # Calculate different position sizes
        sizes = {
            'fixed_fractional': self._fixed_fractional_size(entry_price, stop_loss),
            'kelly': self._kelly_criterion_size(signal, market_data),
            'volatility_based': self._volatility_based_size(symbol, market_data),
            'optimal_f': self._optimal_f_size(signal, portfolio_state),
            'risk_parity': self._risk_parity_size(symbol, portfolio_state),
            'machine_learning': self._ml_based_size(signal, market_data, portfolio_state),
            'regime_adjusted': self._regime_adjusted_size(signal, market_data, portfolio_state)
        }
        
        # Weight different methods based on market conditions
        weights = self._calculate_method_weights(market_data, portfolio_state)
        
        # Calculate weighted position size
        weighted_size = sum(
            sizes[method] * weight 
            for method, weight in weights.items() 
            if method in sizes
        )
        
        # Apply adjustments
        adjusted_size = self._apply_adjustments(
            weighted_size, signal, market_data, portfolio_state
        )
        
        # Apply limits
        final_size = self._apply_limits(adjusted_size, entry_price)
        
        # Log sizing decision
        self._log_sizing_decision(signal, sizes, weights, final_size)
        
        return final_size
    
    def _fixed_fractional_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Fixed fractional position sizing.
        
        Args:
            entry_price: Entry price for position
            stop_loss: Stop loss price
            
        Returns:
            Position size in base currency
        """
        available_capital = self.risk_manager.get_available_capital()
        risk_per_trade = self.risk_manager.risk_params['risk_per_trade']
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        # Position size in units
        position_units = (available_capital * risk_per_trade) / risk_per_unit
        
        # Convert to dollar size
        position_size = position_units * entry_price
        
        return position_size
    
    def _kelly_criterion_size(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, float]
    ) -> float:
        """
        Kelly criterion position sizing.
        
        Kelly formula: f = (p*b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        
        Args:
            signal: Trading signal
            market_data: Market conditions
            
        Returns:
            Position size based on Kelly criterion
        """
        # Estimate win probability and win/loss ratio
        win_prob = signal.get('win_probability', 0.55)
        
        # Calculate expected win/loss amounts
        take_profit = signal.get('take_profit', signal['entry_price'] * 1.03)
        stop_loss = signal.get('stop_loss', signal['entry_price'] * 0.98)
        entry_price = signal['entry_price']
        
        win_amount = abs(take_profit - entry_price) / entry_price
        loss_amount = abs(entry_price - stop_loss) / entry_price
        
        # Kelly formula
        if loss_amount > 0:
            b = win_amount / loss_amount
            q = 1 - win_prob
            
            kelly_fraction = (win_prob * b - q) / b
            
            # Apply safety factor
            kelly_fraction *= self.params['kelly_fraction']
            
            # Ensure non-negative
            kelly_fraction = max(0, kelly_fraction)
            
            # Convert to position size
            available_capital = self.risk_manager.get_available_capital()
            position_size = available_capital * kelly_fraction
            
            return position_size
        
        return 0
    
    def _volatility_based_size(
        self, 
        symbol: str, 
        market_data: Dict[str, float]
    ) -> float:
        """
        Volatility-based position sizing.
        
        Size positions to target consistent portfolio volatility.
        
        Args:
            symbol: Trading symbol
            market_data: Market conditions
            
        Returns:
            Volatility-adjusted position size
        """
        # Get volatility
        volatility = market_data.get('volatility', 0.02)
        
        # Target volatility (e.g., 1% portfolio volatility)
        target_volatility = 0.01
        
        # Calculate position size to achieve target volatility
        available_capital = self.risk_manager.get_available_capital()
        
        if volatility > 0:
            volatility_ratio = target_volatility / volatility
            position_size = available_capital * volatility_ratio * 0.1  # 10% base allocation
        else:
            position_size = available_capital * 0.05  # Default 5%
        
        return position_size
    
    def _optimal_f_size(
        self, 
        signal: Dict[str, Any], 
        portfolio_state: Dict[str, Any]
    ) -> float:
        """
        Optimal f position sizing based on historical performance.
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            
        Returns:
            Optimal f based position size
        """
        # Get historical trade results for similar signals
        historical_results = self._get_historical_results(signal)
        
        if len(historical_results) < 20:
            # Not enough history, use default
            return self._fixed_fractional_size(
                signal['entry_price'], 
                signal.get('stop_loss', signal['entry_price'] * 0.98)
            )
        
        # Calculate optimal f
        returns = np.array(historical_results)
        
        # Grid search for optimal f
        f_values = np.linspace(0.01, 0.5, 50)
        twrs = []
        
        for f in f_values:
            twr = self._calculate_twr(returns, f)
            twrs.append(twr)
        
        # Find optimal f
        optimal_idx = np.argmax(twrs)
        optimal_f = f_values[optimal_idx]
        
        # Apply safety factor
        optimal_f *= 0.5
        
        # Convert to position size
        available_capital = self.risk_manager.get_available_capital()
        position_size = available_capital * optimal_f
        
        return position_size
    
    def _risk_parity_size(
        self, 
        symbol: str, 
        portfolio_state: Dict[str, Any]
    ) -> float:
        """
        Risk parity position sizing.
        
        Size positions so each contributes equally to portfolio risk.
        
        Args:
            symbol: Trading symbol
            portfolio_state: Current portfolio state
            
        Returns:
            Risk parity position size
        """
        # Get current positions
        positions = portfolio_state.get('positions', {})
        
        if not positions:
            # First position, use default allocation
            available_capital = self.risk_manager.get_available_capital()
            return available_capital * 0.1
        
        # Calculate risk contribution of each position
        risk_contributions = {}
        total_risk = 0
        
        for sym, pos in positions.items():
            volatility = pos.get('volatility', 0.02)
            position_value = pos['size'] * pos.get('current_price', pos['entry_price'])
            risk_contribution = position_value * volatility
            risk_contributions[sym] = risk_contribution
            total_risk += risk_contribution
        
        # Target equal risk contribution
        target_risk = total_risk / (len(positions) + 1)  # +1 for new position
        
        # Calculate position size for target risk
        new_volatility = portfolio_state.get('market_data', {}).get('volatility', 0.02)
        
        if new_volatility > 0:
            position_size = target_risk / new_volatility
        else:
            position_size = self.risk_manager.get_available_capital() * 0.05
        
        return position_size
    
    def _ml_based_size(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, float], 
        portfolio_state: Dict[str, Any]
    ) -> float:
        """
        Machine learning-based position sizing.
        
        Uses features to predict optimal position size.
        
        Args:
            signal: Trading signal
            market_data: Market conditions
            portfolio_state: Portfolio state
            
        Returns:
            ML-predicted position size
        """
        # Extract features for ML model
        features = [
            signal.get('confidence', 0.5),
            market_data.get('volatility', 0.02),
            market_data.get('volume_ratio', 1.0),
            len(portfolio_state.get('positions', {})),
            self.risk_manager.current_drawdown,
            market_data.get('trend_strength', 0.0),
            signal.get('expected_return', 0.02)
        ]
        
        # Simple ML-based sizing (in practice, would use trained model)
        # This is a placeholder implementation
        confidence_factor = signal.get('confidence', 0.5)
        volatility_factor = min(0.02 / market_data.get('volatility', 0.02), 2.0)
        
        base_size = self.risk_manager.get_available_capital() * 0.1
        ml_adjusted_size = base_size * confidence_factor * volatility_factor
        
        return ml_adjusted_size
    
    def _regime_adjusted_size(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, float], 
        portfolio_state: Dict[str, Any]
    ) -> float:
        """
        Regime-adjusted position sizing.
        
        Adjusts size based on detected market regime.
        
        Args:
            signal: Trading signal
            market_data: Market conditions
            portfolio_state: Portfolio state
            
        Returns:
            Regime-adjusted position size
        """
        regime = market_data.get('regime', 'normal')
        
        # Regime-based size adjustments
        regime_multipliers = {
            'low_volatility': 1.5,
            'normal': 1.0,
            'high_volatility': 0.6,
            'extreme_volatility': 0.3,
            'crisis': 0.1
        }
        
        base_size = self._fixed_fractional_size(
            signal['entry_price'], 
            signal.get('stop_loss', signal['entry_price'] * 0.98)
        )
        
        multiplier = regime_multipliers.get(regime, 1.0)
        
        # Additional adjustment for regime confidence
        regime_confidence = market_data.get('regime_confidence', 0.7)
        if regime_confidence < 0.5:
            multiplier = (multiplier + 1.0) / 2  # Average with neutral sizing
        
        return base_size * multiplier
    
    def _calculate_method_weights(
        self, 
        market_data: Dict[str, float], 
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate weights for different sizing methods based on market conditions.
        
        Args:
            market_data: Market conditions
            portfolio_state: Portfolio state
            
        Returns:
            Dictionary of method weights
        """
        weights = {
            'fixed_fractional': 0.2,
            'kelly': 0.15,
            'volatility_based': 0.15,
            'optimal_f': 0.1,
            'risk_parity': 0.1,
            'machine_learning': 0.15,
            'regime_adjusted': 0.15
        }
        
        # Adjust based on market regime
        regime = market_data.get('regime', 'normal')
        
        if regime == 'high_volatility':
            # Favor regime-adjusted and volatility-based in high volatility
            weights['regime_adjusted'] = 0.3
            weights['volatility_based'] = 0.25
            weights['fixed_fractional'] = 0.2
            weights['machine_learning'] = 0.15
            weights['kelly'] = 0.05
            weights['optimal_f'] = 0.025
            weights['risk_parity'] = 0.025
            
        elif regime == 'low_volatility':
            # Favor Kelly and machine learning in low volatility
            weights['kelly'] = 0.25
            weights['machine_learning'] = 0.25
            weights['optimal_f'] = 0.2
            weights['regime_adjusted'] = 0.15
            weights['volatility_based'] = 0.1
            weights['fixed_fractional'] = 0.05
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _apply_adjustments(
        self, 
        base_size: float, 
        signal: Dict[str, Any], 
        market_data: Dict[str, float], 
        portfolio_state: Dict[str, Any]
    ) -> float:
        """
        Apply various adjustments to base position size.
        
        Args:
            base_size: Base position size
            signal: Trading signal
            market_data: Market conditions
            portfolio_state: Portfolio state
            
        Returns:
            Adjusted position size
        """
        adjusted_size = base_size
        
        # Confidence adjustment
        confidence = signal.get('confidence', 0.5)
        if confidence < self.params['confidence_threshold']:
            confidence_factor = confidence / self.params['confidence_threshold']
            adjusted_size *= confidence_factor
        
        # Regime adjustment
        regime = market_data.get('regime', 'normal')
        regime_factor = self.params['regime_adjustments'].get(regime, 1.0)
        adjusted_size *= regime_factor
        
        # Correlation adjustment
        correlation_factor = self._calculate_correlation_adjustment(
            signal['symbol'], 
            portfolio_state
        )
        adjusted_size *= correlation_factor
        
        # Drawdown adjustment
        current_drawdown = self.risk_manager.current_drawdown
        if current_drawdown > 0.1:
            drawdown_factor = 1 - (current_drawdown - 0.1) * 2  # Reduce by 20% per 10% drawdown
            drawdown_factor = max(0.3, drawdown_factor)  # Minimum 30% of original
            adjusted_size *= drawdown_factor
        
        # Time of day adjustment (crypto markets can be less liquid at certain times)
        hour = pd.Timestamp.now().hour
        if hour >= 0 and hour < 6:  # Late night US time
            adjusted_size *= 0.8
        
        return adjusted_size
    
    def _calculate_correlation_adjustment(
        self, 
        symbol: str, 
        portfolio_state: Dict[str, Any]
    ) -> float:
        """
        Calculate position size adjustment based on correlation with existing positions.
        
        Args:
            symbol: Trading symbol
            portfolio_state: Portfolio state
            
        Returns:
            Correlation adjustment factor
        """
        positions = portfolio_state.get('positions', {})
        
        if not positions:
            return 1.0
        
        # Calculate average correlation
        correlations = []
        for existing_symbol in positions:
            correlation = self.risk_manager._get_pair_correlation(symbol, existing_symbol)
            correlations.append(correlation)
        
        avg_correlation = np.mean(correlations)
        
        # Apply penalty for high correlation
        if avg_correlation > 0.7:
            return 1 - self.params['correlation_penalty']
        elif avg_correlation > 0.5:
            return 1 - self.params['correlation_penalty'] * 0.5
        else:
            return 1.0
    
    def _apply_limits(self, size: float, entry_price: float) -> float:
        """
        Apply position size limits.
        
        Args:
            size: Calculated position size
            entry_price: Entry price
            
        Returns:
            Limited position size
        """
        # Maximum position size
        available_capital = self.risk_manager.get_available_capital()
        max_size = available_capital * self.params['max_position_pct']
        
        size = min(size, max_size)
        
        # Minimum position size
        size = max(size, self.params['min_position_size'])
        
        # Round to reasonable precision
        position_units = size / entry_price
        position_units = round(position_units, 8)  # 8 decimal places for crypto
        
        final_size = position_units * entry_price
        
        return final_size
    
    def _get_historical_results(self, signal: Dict[str, Any]) -> List[float]:
        """
        Get historical results for similar signals.
        
        Args:
            signal: Trading signal
            
        Returns:
            List of historical returns
        """
        # In practice, this would query a database of historical trades
        # For now, return simulated results
        
        # Simulate based on signal confidence
        confidence = signal.get('confidence', 0.5)
        win_rate = 0.4 + confidence * 0.3  # 40-70% win rate based on confidence
        
        results = []
        for _ in range(50):
            if np.random.random() < win_rate:
                # Win
                result = np.random.uniform(0.01, 0.05)  # 1-5% win
            else:
                # Loss
                result = np.random.uniform(-0.03, -0.01)  # 1-3% loss
            results.append(result)
        
        return results
    
    def _calculate_twr(self, returns: np.ndarray, f: float) -> float:
        """
        Calculate Terminal Wealth Relative for optimal f.
        
        Args:
            returns: Array of returns
            f: Fraction to test
            
        Returns:
            Terminal wealth relative
        """
        twr = 1.0
        for r in returns:
            twr *= (1 + f * r)
            if twr <= 0:
                return 0
        return twr
    
    def _log_sizing_decision(
        self, 
        signal: Dict[str, Any], 
        sizes: Dict[str, float], 
        weights: Dict[str, float], 
        final_size: float
    ) -> None:
        """
        Log position sizing decision for analysis.
        
        Args:
            signal: Trading signal
            sizes: Calculated sizes by method
            weights: Method weights
            final_size: Final position size
        """
        decision = {
            'timestamp': pd.Timestamp.now(),
            'symbol': signal['symbol'],
            'signal_confidence': signal.get('confidence', 0.5),
            'sizes': sizes,
            'weights': weights,
            'final_size': final_size,
            'available_capital': self.risk_manager.get_available_capital()
        }
        
        self.sizing_history.append(decision)
        
        logger.info(f"Position sizing for {signal['symbol']}: ${final_size:.2f}")
        logger.debug(f"Sizing details: {sizes}")
    
    def get_sizing_analytics(self) -> Dict[str, Any]:
        """
        Get analytics on position sizing decisions.
        
        Returns:
            Dictionary of sizing analytics
        """
        if not self.sizing_history:
            return {}
        
        df = pd.DataFrame(self.sizing_history)
        
        analytics = {
            'avg_position_size': df['final_size'].mean(),
            'std_position_size': df['final_size'].std(),
            'avg_confidence': df['signal_confidence'].mean(),
            'position_size_by_confidence': df.groupby(
                pd.cut(df['signal_confidence'], bins=5)
            )['final_size'].mean().to_dict(),
            'method_usage': {
                method: df['weights'].apply(lambda x: x.get(method, 0)).mean()
                for method in [
                    'fixed_fractional', 'kelly', 'volatility_based', 
                    'optimal_f', 'risk_parity', 'machine_learning', 'regime_adjusted'
                ]
            }
        }
        
        return analytics
    
    def optimize_parameters(self, historical_data: pd.DataFrame) -> None:
        """
        Optimize position sizing parameters based on historical data.
        
        Args:
            historical_data: Historical trading data
        """
        # This would implement parameter optimization
        # For now, log that it should be done
        logger.info("Position sizing parameter optimization should be performed periodically")