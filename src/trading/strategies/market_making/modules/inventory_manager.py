"""
Inventory management module for market making strategy.
Handles position tracking, inventory skew calculation, and rebalancing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class InventoryManager:
    """Manages inventory positions and generates rebalancing recommendations"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.inventory = {}
        self.filled_orders = deque(maxlen=1000)
        
    def update_inventory(self, fill_data: Dict) -> bool:
        """Update inventory based on order fill"""
        try:
            # Validate fill data
            required_fields = ['symbol', 'side', 'size', 'price']
            for field in required_fields:
                if field not in fill_data:
                    logger.error(f"Missing required field in fill data: {field}")
                    return False
            
            symbol = fill_data['symbol']
            side = fill_data['side']
            size = float(fill_data['size'])
            price = float(fill_data['price'])
            
            # Validate values
            if size <= 0 or price <= 0:
                logger.error(f"Invalid fill values: size={size}, price={price}")
                return False
            
            # Initialize inventory if needed
            if symbol not in self.inventory:
                self.inventory[symbol] = 0
            
            old_inventory = self.inventory[symbol]
            
            # Update inventory with overflow protection
            if side == 'buy':
                self.inventory[symbol] = min(
                    self.inventory[symbol] + size,
                    self.params['max_inventory'] / price  # Max position in units
                )
            else:
                self.inventory[symbol] = max(
                    self.inventory[symbol] - size,
                    -self.params['max_inventory'] / price  # Max short position
                )
            
            # Check for inventory overflow
            if abs(self.inventory[symbol] - old_inventory) < size * 0.99:
                logger.warning(f"Inventory limit reached for {symbol}")
            
            # Track fill
            self.filled_orders.append({
                'timestamp': pd.Timestamp.now(),
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'fee': float(fill_data.get('fee', 0)),
                'inventory_after': self.inventory[symbol]
            })
            
            logger.info(f"Inventory updated: {symbol} {side} {size} @ {price}, "
                       f"inventory: {self.inventory[symbol]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating inventory: {e}")
            return False
    
    def calculate_inventory_skew(self, symbol: str, current_price: float) -> float:
        """Calculate inventory skew factor for quote adjustment"""
        try:
            if symbol not in self.inventory:
                return 0
            
            inventory_value = self.inventory[symbol] * current_price
            max_inventory = self.params['max_inventory']
            
            if max_inventory <= 0:
                return 0
            
            # Normalize inventory to [-1, 1]
            skew = inventory_value / max_inventory
            skew = max(-1, min(1, skew))
            
            # Apply non-linear transformation for stronger effect near limits
            if abs(skew) > 0.8:
                # Amplify skew when approaching limits
                skew = np.sign(skew) * (0.8 + (abs(skew) - 0.8) * 2)
                skew = max(-1, min(1, skew))
            else:
                # Normal square root transformation
                skew = np.sign(skew) * (abs(skew) ** 0.5)
            
            return skew
            
        except Exception as e:
            logger.error(f"Error calculating inventory skew: {e}")
            return 0
    
    def check_inventory_limits(self, symbol: str, current_price: float) -> bool:
        """Check if inventory is within acceptable limits"""
        try:
            if symbol not in self.inventory:
                return True
            
            inventory_value = abs(self.inventory[symbol] * current_price)
            return inventory_value <= self.params['max_inventory']
            
        except Exception as e:
            logger.error(f"Error checking inventory limits: {e}")
            return False
    
    def needs_rebalancing(self, symbol: str, current_price: float) -> bool:
        """Check if inventory needs rebalancing"""
        try:
            if symbol not in self.inventory:
                return False
            
            inventory_value = abs(self.inventory[symbol] * current_price)
            threshold_value = self.params['max_inventory'] * self.params['rebalance_threshold']
            
            return inventory_value > threshold_value
            
        except Exception as e:
            logger.error(f"Error checking rebalancing need: {e}")
            return False
    
    def generate_rebalance_orders(self, symbol: str, current_price: float,
                                target_inventory: float = 0) -> List[Dict]:
        """Generate orders to rebalance inventory"""
        try:
            current_inventory = self.inventory.get(symbol, 0)
            imbalance = current_inventory - target_inventory
            
            if abs(imbalance) < self.params['min_order_size'] / current_price:
                return []
            
            orders = []
            
            # Split rebalance into smaller orders
            chunk_value = self.params['min_order_size'] * 10
            chunk_size = chunk_value / current_price
            num_orders = min(5, int(abs(imbalance) / chunk_size) + 1)  # Cap at 5 orders
            
            for i in range(num_orders):
                size = min(chunk_size, abs(imbalance) - i * chunk_size)
                if size < self.params['min_order_size'] / current_price:
                    break
                
                orders.append({
                    'symbol': symbol,
                    'side': 'sell' if imbalance > 0 else 'buy',
                    'size': size,
                    'type': 'limit',
                    'time_in_force': 'GTC',
                    'post_only': False  # May need to cross spread
                })
            
            return orders
            
        except Exception as e:
            logger.error(f"Error generating rebalance orders: {e}")
            return []
    
    def calculate_inventory_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate unrealized PnL for all positions"""
        try:
            pnl_by_symbol = {}
            total_pnl = 0
            
            for symbol, position in self.inventory.items():
                if position == 0:
                    pnl_by_symbol[symbol] = 0
                    continue
                
                # Get current price
                current_price = current_prices.get(symbol, 0)
                if current_price <= 0:
                    current_price = self._estimate_price(symbol)
                
                if current_price <= 0:
                    continue
                
                # Calculate average entry price from recent fills
                avg_price = self._calculate_average_entry_price(symbol)
                
                if avg_price > 0:
                    # PnL calculation
                    if position > 0:
                        pnl = (current_price - avg_price) * position
                    else:
                        pnl = (avg_price - current_price) * abs(position)
                    
                    # Sanity check PnL
                    max_reasonable_pnl = abs(position) * current_price * 0.5  # 50% max
                    if abs(pnl) > max_reasonable_pnl:
                        logger.warning(f"Unreasonable PnL for {symbol}: {pnl}")
                        pnl = 0
                    
                    pnl_by_symbol[symbol] = pnl
                    total_pnl += pnl
                else:
                    pnl_by_symbol[symbol] = 0
            
            pnl_by_symbol['total'] = total_pnl
            return pnl_by_symbol
            
        except Exception as e:
            logger.error(f"Error calculating inventory PnL: {e}")
            return {'total': 0}
    
    def _calculate_average_entry_price(self, symbol: str) -> float:
        """Calculate weighted average entry price for a position"""
        try:
            symbol_fills = [f for f in self.filled_orders if f['symbol'] == symbol]
            
            if not symbol_fills:
                return 0
            
            # Calculate weighted average price
            buy_value = sum(f['price'] * f['size'] for f in symbol_fills if f['side'] == 'buy')
            buy_volume = sum(f['size'] for f in symbol_fills if f['side'] == 'buy')
            sell_value = sum(f['price'] * f['size'] for f in symbol_fills if f['side'] == 'sell')
            sell_volume = sum(f['size'] for f in symbol_fills if f['side'] == 'sell')
            
            net_volume = buy_volume - sell_volume
            
            if abs(net_volume) > 0.001:  # Avoid division by zero
                avg_price = abs(buy_value - sell_value) / abs(net_volume)
                return avg_price
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating average entry price: {e}")
            return 0
    
    def _estimate_price(self, symbol: str) -> float:
        """Estimate current price from recent fills"""
        try:
            # Check recent fills
            recent_fills = [
                f for f in self.filled_orders 
                if f['symbol'] == symbol and 
                (pd.Timestamp.now() - f['timestamp']).seconds < 60
            ]
            
            if recent_fills:
                # Use average of recent fill prices
                return np.mean([f['price'] for f in recent_fills])
            
            # Default fallback
            return 50000 if 'BTC' in symbol else 1000
            
        except Exception as e:
            logger.error(f"Error estimating price: {e}")
            return 0
    
    def get_inventory_metrics(self) -> Dict:
        """Get comprehensive inventory metrics"""
        try:
            total_long_value = 0
            total_short_value = 0
            positions = {}
            
            for symbol, units in self.inventory.items():
                price = self._estimate_price(symbol)
                value = units * price
                
                positions[symbol] = {
                    'units': units,
                    'estimated_value': value,
                    'side': 'long' if units > 0 else 'short' if units < 0 else 'flat'
                }
                
                if units > 0:
                    total_long_value += value
                elif units < 0:
                    total_short_value += abs(value)
            
            return {
                'positions': positions,
                'total_long_value': total_long_value,
                'total_short_value': total_short_value,
                'net_exposure': total_long_value - total_short_value,
                'gross_exposure': total_long_value + total_short_value,
                'utilization': (total_long_value + total_short_value) / self.params['max_inventory']
                    if self.params['max_inventory'] > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting inventory metrics: {e}")
            return {}
    
    def update_spread_captured(self, symbol: str, side: str, price: float, size: float) -> float:
        """Calculate spread captured from a fill"""
        try:
            # Look for recent opposite side fill
            recent_fills = [
                f for f in self.filled_orders 
                if f['symbol'] == symbol and 
                (pd.Timestamp.now() - f['timestamp']).seconds < 300  # 5 minutes
            ]
            
            for fill in reversed(recent_fills):  # Check most recent first
                if fill['side'] != side:
                    # Found opposite side fill
                    if side == 'sell':
                        spread = price - fill['price']
                    else:
                        spread = fill['price'] - price
                    
                    if spread > 0:
                        captured_value = spread * min(fill['size'], size)
                        logger.debug(f"Captured spread: {spread:.4f} on {min(fill['size'], size):.4f} units")
                        return captured_value
                    break
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating spread captured: {e}")
            return 0
    
    def get_recent_fills(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent fills, optionally filtered by symbol"""
        try:
            if symbol:
                fills = [f for f in self.filled_orders if f['symbol'] == symbol]
            else:
                fills = list(self.filled_orders)
            
            # Convert timestamps to strings for serialization
            result = []
            for fill in fills[-limit:]:
                fill_copy = fill.copy()
                if 'timestamp' in fill_copy and isinstance(fill_copy['timestamp'], pd.Timestamp):
                    fill_copy['timestamp'] = fill_copy['timestamp'].isoformat()
                result.append(fill_copy)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting recent fills: {e}")
            return []