import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from scipy import stats
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class RiskMetrics:
    total_exposure: float
    position_count: int
    largest_position: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    current_drawdown: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    sharpe_ratio: float
    risk_score: float

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        
        # Risk parameters
        self.risk_params = {
            'max_position_size': 0.1,  # 10% of capital per position
            'max_total_exposure': 0.8,  # 80% total exposure
            'max_correlation': 0.7,  # Maximum correlation between positions
            'stop_loss_pct': 0.02,  # 2% stop loss
            'max_daily_loss': 0.05,  # 5% daily loss limit
            'max_drawdown': 0.20,  # 20% maximum drawdown
            'position_limit': 10,  # Maximum number of positions
            'risk_per_trade': 0.02,  # 2% risk per trade
            'var_confidence': 0.95,  # 95% VaR
            'liquidity_buffer': 0.2,  # 20% liquidity buffer
        }
        
        # Risk tracking
        self.daily_pnl = []
        self.equity_curve = [initial_capital]
        self.high_water_mark = initial_capital
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.daily_trades = 0
        self.last_reset = pd.Timestamp.now()
        
        # Correlation matrix
        self.correlation_matrix = pd.DataFrame()
        
        # Risk limits breached
        self.risk_breaches = []
        
    def check_pre_trade_risk(self, symbol: str, side: str, size: float, 
                           price: float, stop_loss: Optional[float] = None) -> Tuple[bool, str]:
        """Check if trade passes risk criteria"""
        # Check daily loss limit
        if self._check_daily_loss_limit():
            return False, "Daily loss limit reached"
        
        # Check position limit
        if len(self.positions) >= self.risk_params['position_limit'] and symbol not in self.positions:
            return False, "Position limit reached"
        
        # Check position size
        position_value = size * price
        if position_value > self.current_capital * self.risk_params['max_position_size']:
            return False, "Position size too large"
        
        # Check total exposure
        current_exposure = self._calculate_total_exposure()
        new_exposure = current_exposure + position_value
        
        if new_exposure > self.current_capital * self.risk_params['max_total_exposure']:
            return False, "Total exposure limit exceeded"
        
        # Check correlation risk
        if not self._check_correlation_risk(symbol):
            return False, "Correlation risk too high"
        
        # Check liquidity
        if self.current_capital - position_value < self.initial_capital * self.risk_params['liquidity_buffer']:
            return False, "Insufficient liquidity buffer"
        
        # Check stop loss risk
        if stop_loss:
            risk_amount = abs(price - stop_loss) * size
            if risk_amount > self.current_capital * self.risk_params['risk_per_trade']:
                return False, "Stop loss risk too high"
        
        return True, "Risk check passed"
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """Calculate position size based on risk parameters"""
        # Risk amount
        risk_amount = self.current_capital * self.risk_params['risk_per_trade']
        
        # Price risk per unit
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        # Base position size
        position_size = risk_amount / price_risk
        
        # Apply position size limit
        max_position_value = self.current_capital * self.risk_params['max_position_size']
        max_position_size = max_position_value / entry_price
        
        position_size = min(position_size, max_position_size)
        
        # Adjust for correlation risk
        correlation_factor = self._get_correlation_adjustment(symbol)
        position_size *= correlation_factor
        
        # Adjust for current drawdown
        drawdown_factor = self._get_drawdown_adjustment()
        position_size *= drawdown_factor
        
        # Adjust for volatility
        volatility_factor = self._get_volatility_adjustment(symbol)
        position_size *= volatility_factor
        
        return max(0, position_size)
    
    def add_position(self, symbol: str, side: str, size: float, 
                    entry_price: float, stop_loss: Optional[float] = None):
        """Add new position to portfolio"""
        position = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'entry_time': pd.Timestamp.now(),
            'stop_loss': stop_loss or (entry_price * 0.98 if side == 'long' else entry_price * 1.02),
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'unrealized_pnl': 0,
            'realized_pnl': 0
        }
        
        if symbol in self.positions:
            # Update existing position (averaging)
            self._update_position(symbol, position)
        else:
            self.positions[symbol] = position
        
        # Update trade history
        self.trade_history.append({
            'timestamp': position['entry_time'],
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': entry_price,
            'action': 'open'
        })
        
        self.daily_trades += 1
        logger.info(f"Added position: {symbol} {side} {size} @ {entry_price}")
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update position with current price"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Update highest/lowest
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Calculate unrealized PnL
        if position['side'] == 'long':
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['size']
        else:
            position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['size']
        
        # Check stop loss
        if self._check_stop_loss(position, current_price):
            self.close_position(symbol, current_price, "Stop loss triggered")
        
        # Update trailing stop
        self._update_trailing_stop(position, current_price)
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual"):
        """Close position and record PnL"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate realized PnL
        if position['side'] == 'long':
            realized_pnl = (exit_price - position['entry_price']) * position['size']
        else:
            realized_pnl = (position['entry_price'] - exit_price) * position['size']
        
        position['realized_pnl'] = realized_pnl
        
        # Update capital
        self.current_capital += realized_pnl
        
        # Update trade history
        self.trade_history.append({
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'side': 'sell' if position['side'] == 'long' else 'buy',
            'size': position['size'],
            'price': exit_price,
            'action': 'close',
            'pnl': realized_pnl,
            'reason': reason
        })
        
        # Update daily PnL
        self._update_daily_pnl(realized_pnl)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} @ {exit_price}, PnL: {realized_pnl:.2f}, Reason: {reason}")
    
    def _check_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        if position['side'] == 'long':
            return current_price <= position['stop_loss']
        else:
            return current_price >= position['stop_loss']
    
    def _update_trailing_stop(self, position: Dict, current_price: float):
        """Update trailing stop loss"""
        if position['side'] == 'long':
            # For long positions, trail stop up
            if current_price > position['highest_price'] * 0.98:  # 2% trailing
                new_stop = current_price * 0.98
                position['stop_loss'] = max(position['stop_loss'], new_stop)
        else:
            # For short positions, trail stop down
            if current_price < position['lowest_price'] * 1.02:  # 2% trailing
                new_stop = current_price * 1.02
                position['stop_loss'] = min(position['stop_loss'], new_stop)
    
    def _update_position(self, symbol: str, new_position: Dict):
        """Update existing position with new trade"""
        existing = self.positions[symbol]
        
        # Same side - add to position
        if existing['side'] == new_position['side']:
            total_size = existing['size'] + new_position['size']
            avg_price = (existing['entry_price'] * existing['size'] + 
                        new_position['entry_price'] * new_position['size']) / total_size
            
            existing['size'] = total_size
            existing['entry_price'] = avg_price
            existing['stop_loss'] = new_position['stop_loss']  # Use new stop loss
        else:
            # Opposite side - reduce or flip position
            if new_position['size'] > existing['size']:
                # Flip position
                remaining_size = new_position['size'] - existing['size']
                
                # Calculate PnL on closed portion
                if existing['side'] == 'long':
                    closed_pnl = (new_position['entry_price'] - existing['entry_price']) * existing['size']
                else:
                    closed_pnl = (existing['entry_price'] - new_position['entry_price']) * existing['size']
                
                self.current_capital += closed_pnl
                self._update_daily_pnl(closed_pnl)
                
                # Update to new position
                existing['side'] = new_position['side']
                existing['size'] = remaining_size
                existing['entry_price'] = new_position['entry_price']
                existing['stop_loss'] = new_position['stop_loss']
                existing['highest_price'] = new_position['entry_price']
                existing['lowest_price'] = new_position['entry_price']
            else:
                # Reduce position
                existing['size'] -= new_position['size']
                
                # Calculate PnL on closed portion
                if existing['side'] == 'long':
                    closed_pnl = (new_position['entry_price'] - existing['entry_price']) * new_position['size']
                else:
                    closed_pnl = (existing['entry_price'] - new_position['entry_price']) * new_position['size']
                
                self.current_capital += closed_pnl
                self._update_daily_pnl(closed_pnl)
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        total = 0
        for position in self.positions.values():
            # Use current market price if available
            price = position.get('current_price', position['entry_price'])
            total += position['size'] * price
        return total
    
    def _check_correlation_risk(self, symbol: str) -> bool:
        """Check if adding position increases correlation risk"""
        if len(self.positions) == 0:
            return True
        
        # Get correlation with existing positions
        # This is simplified - in practice would use actual correlation data
        high_correlation_count = 0
        
        for existing_symbol in self.positions:
            correlation = self._get_pair_correlation(symbol, existing_symbol)
            if correlation > self.risk_params['max_correlation']:
                high_correlation_count += 1
        
        # Allow if less than half of positions are highly correlated
        return high_correlation_count < len(self.positions) / 2
    
    def _get_pair_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        # In practice, this would use historical correlation matrix
        # For now, return a placeholder
        if symbol1 == symbol2:
            return 1.0
        
        # Assume some correlation for crypto pairs
        return 0.5
    
    def _get_correlation_adjustment(self, symbol: str) -> float:
        """Get position size adjustment based on correlation"""
        if len(self.positions) == 0:
            return 1.0
        
        # Calculate average correlation with existing positions
        correlations = []
        for existing_symbol in self.positions:
            corr = self._get_pair_correlation(symbol, existing_symbol)
            correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        # Reduce size for high correlation
        if avg_correlation > 0.7:
            return 0.5
        elif avg_correlation > 0.5:
            return 0.75
        else:
            return 1.0
    
    def _get_drawdown_adjustment(self) -> float:
        """Adjust position size based on current drawdown"""
        if self.current_drawdown < 0.05:
            return 1.0
        elif self.current_drawdown < 0.10:
            return 0.75
        elif self.current_drawdown < 0.15:
            return 0.5
        else:
            return 0.25
    
    def _get_volatility_adjustment(self, symbol: str) -> float:
        """Adjust position size based on volatility"""
        # In practice, would use actual volatility data
        # For now, return a placeholder
        return 0.8
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is reached"""
        if not self.daily_pnl:
            return False
        
        daily_loss = sum(pnl for pnl in self.daily_pnl if pnl < 0)
        daily_loss_pct = abs(daily_loss) / self.initial_capital
        
        return daily_loss_pct >= self.risk_params['max_daily_loss']
    
    def _update_daily_pnl(self, pnl: float):
        """Update daily PnL tracking"""
        self.daily_pnl.append(pnl)
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        
        # Update high water mark and drawdown
        if self.current_capital > self.high_water_mark:
            self.high_water_mark = self.current_capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.high_water_mark - self.current_capital) / self.high_water_mark
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def calculate_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate Value at Risk"""
        if len(self.daily_pnl) < 20:
            return 0
        
        # Historical VaR
        returns = pd.Series(self.daily_pnl) / self.initial_capital
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile) * self.current_capital * np.sqrt(horizon)
        
        return abs(var)
    
    def calculate_monte_carlo_var(self, confidence: float = 0.95, horizon: int = 1, simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR simulation (NEW - HIGH IMPACT)"""
        if len(self.daily_pnl) < 20:
            return 0
        
        returns = np.array(self.daily_pnl) / self.initial_capital
        
        # Generate random scenarios
        simulated_returns = np.random.choice(returns, size=(simulations, horizon), replace=True)
        portfolio_returns = np.sum(simulated_returns, axis=1)
        
        # Calculate VaR
        var_percentile = (1 - confidence) * 100
        monte_carlo_var = np.percentile(portfolio_returns, var_percentile) * self.current_capital
        
        return abs(monte_carlo_var)
    
    def calculate_parametric_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate Parametric VaR using normal distribution assumption (NEW)"""
        if len(self.daily_pnl) < 20:
            return 0
        
        returns = pd.Series(self.daily_pnl) / self.initial_capital
        
        # Calculate parameters
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Normal distribution critical value
        z_score = stats.norm.ppf(1 - confidence)
        
        # Parametric VaR
        var = (mean_return + z_score * volatility) * self.current_capital * np.sqrt(horizon)
        
        return abs(var)
    
    def calculate_cvar(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(self.daily_pnl) < 20:
            return 0
        
        # Get returns below VaR threshold
        returns = pd.Series(self.daily_pnl) / self.initial_capital
        var_percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, var_percentile)
        
        # Calculate average of returns below threshold
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) > 0:
            cvar = tail_returns.mean() * self.current_capital * np.sqrt(horizon)
            return abs(cvar)
        
        return self.calculate_var(confidence, horizon)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.daily_pnl) < 20:
            return 0
        
        returns = pd.Series(self.daily_pnl) / self.initial_capital
        
        # Annualized return
        daily_return = returns.mean()
        annual_return = daily_return * 252
        
        # Annualized volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        if annual_vol == 0:
            return 0
        
        sharpe = (annual_return - risk_free_rate) / annual_vol
        return sharpe
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        total_exposure = self._calculate_total_exposure()
        position_count = len(self.positions)
        
        # Largest position
        largest_position = 0
        if self.positions:
            largest_position = max(
                pos['size'] * pos.get('current_price', pos['entry_price'])
                for pos in self.positions.values()
            )
        
        # PnL calculations
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        realized_pnl = self.current_capital - self.initial_capital - unrealized_pnl
        total_pnl = realized_pnl + unrealized_pnl
        
        # Risk metrics - Enhanced with multiple VaR methods
        var_95_historical = self.calculate_var(0.95)
        var_95_monte_carlo = self.calculate_monte_carlo_var(0.95)
        var_95_parametric = self.calculate_parametric_var(0.95)
        
        # Use the most conservative (highest) VaR estimate
        var_95 = max(var_95_historical, var_95_monte_carlo, var_95_parametric)
        cvar_95 = self.calculate_cvar(0.95)
        sharpe = self.calculate_sharpe_ratio()
        
        # Overall risk score (0-100, higher is riskier)
        risk_score = self._calculate_risk_score(
            total_exposure, position_count, self.current_drawdown, var_95
        )
        
        return RiskMetrics(
            total_exposure=total_exposure,
            position_count=position_count,
            largest_position=largest_position,
            total_pnl=total_pnl,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(self, exposure: float, positions: int, 
                            drawdown: float, var: float) -> float:
        """Calculate overall risk score"""
        # Exposure risk (0-30 points)
        exposure_pct = exposure / self.current_capital
        exposure_score = min(30, exposure_pct * 30 / self.risk_params['max_total_exposure'])
        
        # Position concentration risk (0-20 points)
        concentration_score = min(20, positions * 20 / self.risk_params['position_limit'])
        
        # Drawdown risk (0-30 points)
        drawdown_score = min(30, drawdown * 30 / self.risk_params['max_drawdown'])
        
        # VaR risk (0-20 points)
        var_pct = var / self.current_capital
        var_score = min(20, var_pct * 20 / 0.05)  # 5% VaR as baseline
        
        total_score = exposure_score + concentration_score + drawdown_score + var_score
        
        return min(100, total_score)
    
    def reset_daily_counters(self):
        """Reset daily counters and limits"""
        current_time = pd.Timestamp.now()
        
        # Reset if new day
        if current_time.date() > self.last_reset.date():
            self.daily_pnl = []
            self.daily_trades = 0
            self.last_reset = current_time
            logger.info("Daily risk counters reset")
    
    def get_risk_adjusted_leverage(self) -> float:
        """Calculate appropriate leverage based on current risk"""
        base_leverage = 3.0  # Base leverage
        
        # Adjust for drawdown
        if self.current_drawdown > 0.1:
            base_leverage *= 0.5
        elif self.current_drawdown > 0.05:
            base_leverage *= 0.75
        
        # Adjust for volatility (would use actual market volatility)
        # For now, use a placeholder
        volatility_adjustment = 0.8
        
        # Adjust for correlation risk
        if len(self.positions) > 5:
            correlation_adjustment = 0.7
        else:
            correlation_adjustment = 1.0
        
        final_leverage = base_leverage * volatility_adjustment * correlation_adjustment
        
        return max(1.0, min(5.0, final_leverage))  # Cap between 1x and 5x
    
    def export_risk_report(self) -> Dict:
        """Export comprehensive risk report"""
        metrics = self.calculate_risk_metrics()
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'account': {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_pnl': metrics.total_pnl,
                'total_pnl_pct': (metrics.total_pnl / self.initial_capital) * 100
            },
            'positions': {
                'count': metrics.position_count,
                'total_exposure': metrics.total_exposure,
                'exposure_pct': (metrics.total_exposure / self.current_capital) * 100,
                'largest_position': metrics.largest_position,
                'unrealized_pnl': metrics.unrealized_pnl
            },
            'risk_metrics': {
                'current_drawdown': metrics.current_drawdown * 100,
                'max_drawdown': metrics.max_drawdown * 100,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'sharpe_ratio': metrics.sharpe_ratio,
                'risk_score': metrics.risk_score
            },
            'limits': {
                'daily_loss_limit': self.risk_params['max_daily_loss'] * 100,
                'max_drawdown_limit': self.risk_params['max_drawdown'] * 100,
                'position_limit': self.risk_params['position_limit'],
                'daily_trades': self.daily_trades
            },
            'risk_breaches': self.risk_breaches[-10:]  # Last 10 breaches
        }
        
        return report
    
    def log_risk_breach(self, breach_type: str, details: str):
        """Log risk limit breach"""
        breach = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'type': breach_type,
            'details': details,
            'risk_score': self.calculate_risk_metrics().risk_score
        }
        
        self.risk_breaches.append(breach)
        logger.warning(f"Risk breach: {breach_type} - {details}")
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        total_exposure = self._calculate_total_exposure()
        reserved_capital = self.initial_capital * self.risk_params['liquidity_buffer']
        
        available = self.current_capital - total_exposure - reserved_capital
        return max(0, available)
