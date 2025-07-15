import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from collections import deque
from ..risk_manager import RiskManager
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class MarketMakingQuote:
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    spread: float
    mid_price: float
    fair_value: float
    inventory_skew: float
    volatility_adjustment: float
    confidence: float

class MarketMakingStrategy:
    """Advanced market making strategy with inventory management"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.active_quotes = {}
        self.inventory = {}
        self.filled_orders = deque(maxlen=1000)
        
        # Strategy parameters
        self.params = {
            'base_spread': 0.002,  # 0.2% base spread
            'min_spread': 0.001,   # 0.1% minimum spread
            'max_spread': 0.01,    # 1% maximum spread
            'inventory_target': 0,  # Target neutral inventory
            'max_inventory': 10000,  # Maximum inventory in USD
            'inventory_skew_factor': 0.5,  # How much to skew quotes based on inventory
            'volatility_factor': 2.0,  # Spread multiplier for volatility
            'order_levels': 3,  # Number of order levels
            'level_spacing': 0.001,  # 0.1% between levels
            'size_decay': 0.5,  # Size reduction per level
            'min_order_size': 10,  # Minimum order size in USD
            'rebalance_threshold': 0.1,  # 10% inventory imbalance triggers rebalance
            'maker_fee': -0.0001,  # -0.01% maker rebate
            'taker_fee': 0.00035   # 0.035% taker fee
        }
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'gross_pnl': 0,
            'fee_revenue': 0,
            'inventory_pnl': 0,
            'spread_captured': 0
        }
        
        # Market microstructure tracking
        self.microstructure = {}
        
    def calculate_quotes(self, market_data: Dict[str, any], 
                        ml_predictions: Optional[Dict] = None) -> List[MarketMakingQuote]:
        """Calculate market making quotes for multiple levels"""
        quotes = []
        
        # Get current market state
        symbol = market_data.get('symbol', 'BTC-USD')
        best_bid = market_data['best_bid']
        best_ask = market_data['best_ask']
        mid_price = (best_bid + best_ask) / 2
        current_spread = best_ask - best_bid
        
        # Calculate fair value
        fair_value = self._calculate_fair_value(market_data, ml_predictions)
        
        # Get inventory position
        current_inventory = self.inventory.get(symbol, 0)
        inventory_value = current_inventory * mid_price
        
        # Calculate inventory skew
        inventory_skew = self._calculate_inventory_skew(inventory_value)
        
        # Calculate volatility adjustment
        volatility = market_data.get('volatility', 0.02)
        volatility_adjustment = self.params['volatility_factor'] * volatility
        
        # Generate quotes for each level
        for level in range(self.params['order_levels']):
            # Calculate spread for this level
            level_spread = self._calculate_level_spread(
                level, current_spread, volatility_adjustment, inventory_skew
            )
            
            # Calculate prices
            bid_price = fair_value - level_spread / 2 - (level * self.params['level_spacing'] * fair_value)
            ask_price = fair_value + level_spread / 2 + (level * self.params['level_spacing'] * fair_value)
            
            # Apply inventory skew
            skew_adjustment = inventory_skew * level_spread * self.params['inventory_skew_factor']
            bid_price -= skew_adjustment
            ask_price -= skew_adjustment
            
            # Calculate sizes
            base_size = self._calculate_base_size(market_data, level)
            bid_size = base_size * (1 + inventory_skew * 0.5)  # Buy more when short
            ask_size = base_size * (1 - inventory_skew * 0.5)  # Sell more when long
            
            # Apply size decay
            size_multiplier = (1 - self.params['size_decay']) ** level
            bid_size *= size_multiplier
            ask_size *= size_multiplier
            
            # Ensure minimum size
            bid_size = max(bid_size, self.params['min_order_size'])
            ask_size = max(ask_size, self.params['min_order_size'])
            
            # Calculate confidence
            confidence = self._calculate_quote_confidence(
                market_data, fair_value, level, ml_predictions
            )
            
            quote = MarketMakingQuote(
                symbol=symbol,
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size,
                spread=ask_price - bid_price,
                mid_price=(bid_price + ask_price) / 2,
                fair_value=fair_value,
                inventory_skew=inventory_skew,
                volatility_adjustment=volatility_adjustment,
                confidence=confidence
            )
            
            quotes.append(quote)
        
        return quotes
    
    def _calculate_fair_value(self, market_data: Dict, ml_predictions: Optional[Dict]) -> float:
        """Calculate fair value using multiple signals"""
        # Start with market mid price
        best_bid = market_data['best_bid']
        best_ask = market_data['best_ask']
        market_mid = (best_bid + best_ask) / 2
        
        fair_value_components = [market_mid]
        weights = [0.4]  # Market mid weight
        
        # Order flow imbalance signal
        if 'order_imbalance' in market_data:
            imbalance = market_data['order_imbalance']
            # Positive imbalance suggests higher fair value
            flow_adjustment = market_mid * imbalance * 0.001
            fair_value_components.append(market_mid + flow_adjustment)
            weights.append(0.2)
        
        # Recent trades signal
        if 'recent_trades' in market_data:
            trades = market_data['recent_trades']
            if trades:
                # VWAP of recent trades
                total_volume = sum(t['size'] for t in trades)
                if total_volume > 0:
                    vwap = sum(t['price'] * t['size'] for t in trades) / total_volume
                    fair_value_components.append(vwap)
                    weights.append(0.2)
        
        # ML prediction signal
        if ml_predictions and 'price_prediction' in ml_predictions:
            ml_price = ml_predictions['price_prediction']
            ml_confidence = ml_predictions.get('confidence', 0.5)
            fair_value_components.append(ml_price)
            weights.append(0.2 * ml_confidence)
        
        # Microprice (weighted by order book sizes)
        if 'bid_volume' in market_data and 'ask_volume' in market_data:
            bid_vol = market_data['bid_volume']
            ask_vol = market_data['ask_volume']
            if bid_vol + ask_vol > 0:
                microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
                fair_value_components.append(microprice)
                weights.append(0.2)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Calculate weighted fair value
        fair_value = sum(comp * weight for comp, weight in zip(fair_value_components, weights))
        
        return fair_value
    
    def _calculate_inventory_skew(self, inventory_value: float) -> float:
        """Calculate inventory skew factor (-1 to 1)"""
        max_inventory = self.params['max_inventory']
        
        # Normalize inventory to [-1, 1]
        skew = inventory_value / max_inventory
        skew = max(-1, min(1, skew))  # Clamp to [-1, 1]
        
        # Apply non-linear transformation for stronger effect near limits
        skew = np.sign(skew) * (abs(skew) ** 0.5)
        
        return skew
    
    def _calculate_level_spread(self, level: int, market_spread: float, 
                               volatility_adj: float, inventory_skew: float) -> float:
        """Calculate spread for a specific level"""
        # Base spread increases with level
        base = self.params['base_spread'] * (1 + level * 0.5)
        
        # Volatility adjustment
        volatility_spread = base * (1 + volatility_adj)
        
        # Competition adjustment (tighter spread if market spread is tight)
        competition_factor = min(1, market_spread / self.params['base_spread'])
        adjusted_spread = volatility_spread * (0.5 + 0.5 * competition_factor)
        
        # Ensure within bounds
        final_spread = max(self.params['min_spread'], 
                          min(self.params['max_spread'], adjusted_spread))
        
        return final_spread
    
    def _calculate_base_size(self, market_data: Dict, level: int) -> float:
        """Calculate base order size"""
        # Base size depends on available capital and risk limits
        available_capital = self.risk_manager.get_available_capital()
        max_position_size = available_capital * 0.1  # 10% per position
        
        # Adjust for market conditions
        volatility = market_data.get('volatility', 0.02)
        volatility_factor = max(0.5, 1 - volatility * 10)  # Reduce size in high volatility
        
        # Level-based sizing
        level_factor = 1 / (level + 1)
        
        base_size = max_position_size * volatility_factor * level_factor
        
        return base_size
    
    def _calculate_quote_confidence(self, market_data: Dict, fair_value: float, 
                                  level: int, ml_predictions: Optional[Dict]) -> float:
        """Calculate confidence in quote"""
        confidence_factors = []
        
        # Fair value confidence (how close to market)
        mid_price = (market_data['best_bid'] + market_data['best_ask']) / 2
        price_deviation = abs(fair_value - mid_price) / mid_price
        price_confidence = max(0, 1 - price_deviation * 50)
        confidence_factors.append(price_confidence)
        
        # Spread confidence (tighter market = higher confidence)
        spread = market_data['best_ask'] - market_data['best_bid']
        spread_pct = spread / mid_price
        spread_confidence = max(0, 1 - spread_pct * 100)
        confidence_factors.append(spread_confidence)
        
        # Level confidence (closer levels = higher confidence)
        level_confidence = 1 / (level + 1)
        confidence_factors.append(level_confidence)
        
        # ML confidence
        if ml_predictions:
            ml_conf = ml_predictions.get('confidence', 0.5)
            confidence_factors.append(ml_conf)
        
        return np.mean(confidence_factors)
    
    async def execute_quotes(self, quotes: List[MarketMakingQuote], executor) -> Dict:
        """Place market making orders"""
        results = {
            'placed_orders': [],
            'failed_orders': [],
            'total_quotes': len(quotes) * 2  # Bid and ask for each level
        }
        
        tasks = []
        
        for quote in quotes:
            # Cancel existing orders at this level
            if quote.symbol in self.active_quotes:
                await self._cancel_existing_quotes(quote.symbol, executor)
            
            # Place bid order
            bid_task = executor.place_order_async(
                symbol=quote.symbol,
                side='buy',
                size=quote.bid_size,
                price=quote.bid_price,
                type='limit',
                time_in_force='GTC',
                post_only=True  # Ensure maker fees
            )
            tasks.append(('bid', quote, bid_task))
            
            # Place ask order
            ask_task = executor.place_order_async(
                symbol=quote.symbol,
                side='sell',
                size=quote.ask_size,
                price=quote.ask_price,
                type='limit',
                time_in_force='GTC',
                post_only=True
            )
            tasks.append(('ask', quote, ask_task))
        
        # Execute all orders
        for order_type, quote, task in tasks:
            try:
                result = await task
                if result['status'] == 'accepted':
                    results['placed_orders'].append({
                        'type': order_type,
                        'quote': quote,
                        'order': result
                    })
                    
                    # Track active quotes
                    if quote.symbol not in self.active_quotes:
                        self.active_quotes[quote.symbol] = []
                    
                    self.active_quotes[quote.symbol].append({
                        'order_id': result['order_id'],
                        'side': order_type,
                        'price': quote.bid_price if order_type == 'bid' else quote.ask_price,
                        'size': quote.bid_size if order_type == 'bid' else quote.ask_size,
                        'level': len([q for q in quotes if q.symbol == quote.symbol])
                    })
                else:
                    results['failed_orders'].append({
                        'type': order_type,
                        'quote': quote,
                        'reason': result.get('reason', 'unknown')
                    })
            except Exception as e:
                logger.error(f"Failed to place {order_type} order: {e}")
                results['failed_orders'].append({
                    'type': order_type,
                    'quote': quote,
                    'error': str(e)
                })
        
        return results
    
    async def _cancel_existing_quotes(self, symbol: str, executor):
        """Cancel existing quotes for a symbol"""
        if symbol in self.active_quotes:
            cancel_tasks = []
            
            for quote in self.active_quotes[symbol]:
                cancel_tasks.append(
                    executor.cancel_order_async(symbol, quote['order_id'])
                )
            
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
            
            # Clear active quotes
            self.active_quotes[symbol] = []
    
    def handle_fill(self, fill_data: Dict):
        """Handle order fill and update inventory"""
        symbol = fill_data['symbol']
        side = fill_data['side']
        size = fill_data['size']
        price = fill_data['price']
        fee = fill_data.get('fee', 0)
        
        # Update inventory
        if symbol not in self.inventory:
            self.inventory[symbol] = 0
        
        if side == 'buy':
            self.inventory[symbol] += size
            self.performance['buy_volume'] += size * price
        else:
            self.inventory[symbol] -= size
            self.performance['sell_volume'] += size * price
        
        # Track fill
        self.filled_orders.append({
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'fee': fee
        })
        
        # Update performance
        self.performance['total_trades'] += 1
        self.performance['fee_revenue'] += fee  # Negative for maker rebates
        
        # Calculate spread captured
        self._update_spread_captured(symbol, side, price)
        
        # Check if rebalancing needed
        if self._needs_rebalancing(symbol):
            logger.info(f"Inventory rebalancing needed for {symbol}")
    
    def _update_spread_captured(self, symbol: str, side: str, price: float):
        """Update spread capture statistics"""
        # Look for recent opposite side fill
        recent_fills = [f for f in self.filled_orders 
                       if f['symbol'] == symbol and 
                       (pd.Timestamp.now() - f['timestamp']).seconds < 300]  # 5 minutes
        
        for fill in recent_fills:
            if fill['side'] != side:
                # Found opposite side fill
                if side == 'sell':
                    spread = price - fill['price']
                else:
                    spread = fill['price'] - price
                
                if spread > 0:
                    self.performance['spread_captured'] += spread * min(fill['size'], price)
                break
    
    def _needs_rebalancing(self, symbol: str) -> bool:
        """Check if inventory needs rebalancing"""
        if symbol not in self.inventory:
            return False
        
        inventory_value = abs(self.inventory[symbol]) * self._get_current_price(symbol)
        
        return inventory_value > self.params['max_inventory'] * self.params['rebalance_threshold']
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        # In practice, this would get the latest price from market data
        return 50000  # Placeholder
    
    def calculate_inventory_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized PnL on inventory"""
        total_pnl = 0
        
        for symbol, position in self.inventory.items():
            if symbol in current_prices and position != 0:
                # Calculate average entry price from recent fills
                symbol_fills = [f for f in self.filled_orders if f['symbol'] == symbol]
                
                if symbol_fills:
                    # Weighted average price
                    total_size = sum(f['size'] for f in symbol_fills)
                    if total_size > 0:
                        avg_price = sum(f['price'] * f['size'] for f in symbol_fills) / total_size
                        current_price = current_prices[symbol]
                        
                        # PnL calculation
                        if position > 0:
                            pnl = (current_price - avg_price) * position
                        else:
                            pnl = (avg_price - current_price) * abs(position)
                        
                        total_pnl += pnl
        
        self.performance['inventory_pnl'] = total_pnl
        return total_pnl
    
    def get_rebalance_orders(self, symbol: str, target_inventory: float = 0) -> List[Dict]:
        """Generate orders to rebalance inventory"""
        current_inventory = self.inventory.get(symbol, 0)
        imbalance = current_inventory - target_inventory
        
        if abs(imbalance) < self.params['min_order_size']:
            return []
        
        orders = []
        
        # Split rebalance into smaller orders to minimize market impact
        chunk_size = self.params['min_order_size'] * 10
        num_orders = int(abs(imbalance) / chunk_size) + 1
        
        for i in range(num_orders):
            size = min(chunk_size, abs(imbalance) - i * chunk_size)
            if size < self.params['min_order_size']:
                break
            
            orders.append({
                'symbol': symbol,
                'side': 'sell' if imbalance > 0 else 'buy',
                'size': size,
                'type': 'limit',
                'time_in_force': 'GTC',
                'post_only': False  # May need to cross spread for rebalancing
            })
        
        return orders
    
    def update_microstructure(self, symbol: str, market_data: Dict):
        """Update market microstructure tracking"""
        if symbol not in self.microstructure:
            self.microstructure[symbol] = {
                'avg_spread': deque(maxlen=100),
                'avg_volume': deque(maxlen=100),
                'volatility': deque(maxlen=100),
                'order_imbalance': deque(maxlen=100)
            }
        
        ms = self.microstructure[symbol]
        
        # Update metrics
        spread = market_data['best_ask'] - market_data['best_bid']
        ms['avg_spread'].append(spread)
        
        if 'volume' in market_data:
            ms['avg_volume'].append(market_data['volume'])
        
        if 'volatility' in market_data:
            ms['volatility'].append(market_data['volatility'])
        
        if 'order_imbalance' in market_data:
            ms['order_imbalance'].append(market_data['order_imbalance'])
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        gross_pnl = self.performance['spread_captured'] + self.performance['fee_revenue']
        
        return {
            'name': 'Market Making',
            'performance': {
                **self.performance,
                'gross_pnl': gross_pnl,
                'net_pnl': gross_pnl + self.performance['inventory_pnl']
            },
            'inventory': self.inventory,
            'active_quotes': sum(len(quotes) for quotes in self.active_quotes.values()),
            'parameters': self.params
        }
