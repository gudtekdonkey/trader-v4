"""Triangular arbitrage opportunity detection and execution"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
import traceback
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ArbitrageOpportunity:
    type: str  # 'triangular', 'statistical', 'funding'
    symbols: List[str]
    entry_prices: Dict[str, float]
    sizes: Dict[str, float]
    expected_profit: float
    expected_profit_pct: float
    confidence: float
    execution_time_estimate: float
    risk_score: float


class TriangularArbitrage:
    """Handles triangular arbitrage detection and calculation"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.failed_attempts = {}
        self.max_failed_attempts = 5
    
    def find_triangular_opportunities(self, market_data: Dict[str, Dict]) -> List[ArbitrageOpportunity]:
        """Find triangular arbitrage opportunities with error handling"""
        opportunities = []
        
        try:
            # Get all available trading pairs
            symbols = list(market_data.keys())
            
            # Limit symbols to prevent combinatorial explosion
            if len(symbols) > 50:
                logger.warning(f"Too many symbols ({len(symbols)}), limiting to 50")
                symbols = symbols[:50]
            
            # Find triangular paths
            for i, symbol1 in enumerate(symbols):
                try:
                    base1, quote1 = self._parse_symbol_safe(symbol1)
                    if not base1 or not quote1:
                        continue
                    
                    for j, symbol2 in enumerate(symbols):
                        if i == j:
                            continue
                        
                        try:
                            base2, quote2 = self._parse_symbol_safe(symbol2)
                            if not base2 or not quote2:
                                continue
                            
                            # Check if we can form a triangle
                            if quote1 == base2:  # e.g., BTC/USDT and ETH/BTC
                                # Look for closing pair
                                closing_symbol = f"{base1}-{quote2}"
                                if closing_symbol in symbols:
                                    # Calculate triangular arbitrage
                                    opp = self._calculate_triangular_arbitrage_safe(
                                        symbol1, symbol2, closing_symbol, market_data
                                    )
                                    
                                    if opp and opp.expected_profit_pct > self.params['min_profit_threshold']:
                                        opportunities.append(opp)
                                        
                        except Exception as e:
                            logger.debug(f"Error checking triangle {symbol1}-{symbol2}: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Error processing {symbol1}: {e}")
                    continue
                    
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in find_triangular_opportunities: {e}")
            return []
    
    def _parse_symbol_safe(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Safely parse symbol with error handling"""
        try:
            # Hyperliquid uses format like "BTC-USD"
            if '-' in symbol:
                parts = symbol.split('-')
                if len(parts) == 2 and all(part.isalpha() for part in parts):
                    return parts[0], parts[1]
            
            # Fallback parsing
            if len(symbol) >= 6:
                return symbol[:3], symbol[3:6]
                
            return None, None
            
        except Exception as e:
            logger.debug(f"Error parsing symbol {symbol}: {e}")
            return None, None
    
    def _calculate_triangular_arbitrage_safe(self, symbol1: str, symbol2: str, symbol3: str,
                                           market_data: Dict[str, Dict]) -> Optional[ArbitrageOpportunity]:
        """Calculate triangular arbitrage with comprehensive error handling"""
        try:
            # Validate all symbols exist
            for symbol in [symbol1, symbol2, symbol3]:
                if symbol not in market_data:
                    return None
            
            # Get best bid/ask for each pair
            data1 = market_data[symbol1]
            data2 = market_data[symbol2]
            data3 = market_data[symbol3]
            
            # Validate prices
            for data in [data1, data2, data3]:
                if not all(key in data for key in ['best_bid', 'best_ask']):
                    return None
                if data['best_bid'] <= 0 or data['best_ask'] <= 0:
                    return None
                if data['best_bid'] >= data['best_ask']:
                    return None
            
            bid1, ask1 = data1['best_bid'], data1['best_ask']
            bid2, ask2 = data2['best_bid'], data2['best_ask']
            bid3, ask3 = data3['best_bid'], data3['best_ask']
            
            # Calculate forward arbitrage with error handling
            try:
                forward_path = 1 / ask1  # Buy base1 with quote1
                forward_path = forward_path * bid2  # Sell base1 for quote2
                forward_path = forward_path * bid3  # Sell base2 for quote1
            except (ZeroDivisionError, OverflowError):
                forward_path = 0
            
            # Calculate reverse arbitrage with error handling
            try:
                reverse_path = 1 / ask3  # Buy base2 with quote1
                reverse_path = reverse_path / ask2  # Buy base1 with base2
                reverse_path = reverse_path * bid1  # Sell base1 for quote1
            except (ZeroDivisionError, OverflowError):
                reverse_path = 0
            
            # Check profitability
            forward_profit = forward_path - 1
            reverse_profit = reverse_path - 1
            
            # Account for fees and slippage
            fee_cost = 0.001 * 3  # 0.1% fee per trade, 3 trades
            slippage_cost = self.params['slippage_buffer'] * 3
            
            forward_profit_net = forward_profit - fee_cost - slippage_cost
            reverse_profit_net = reverse_profit - fee_cost - slippage_cost
            
            # Choose best direction
            if forward_profit_net > reverse_profit_net and forward_profit_net > 0:
                return self._create_triangular_opportunity(
                    'forward', [symbol1, symbol2, symbol3],
                    {'ask1': ask1, 'bid2': bid2, 'bid3': bid3},
                    forward_profit_net, market_data
                )
            elif reverse_profit_net > 0:
                return self._create_triangular_opportunity(
                    'reverse', [symbol3, symbol2, symbol1],
                    {'ask3': ask3, 'ask2': ask2, 'bid1': bid1},
                    reverse_profit_net, market_data
                )
            
        except Exception as e:
            logger.error(f"Error calculating triangular arbitrage: {e}")
            
        return None
    
    def _create_triangular_opportunity(self, direction: str, symbols: List[str],
                                     prices: Dict[str, float], profit: float,
                                     market_data: Dict[str, Dict]) -> ArbitrageOpportunity:
        """Create triangular arbitrage opportunity with validation"""
        try:
            # Map prices to symbols
            if direction == 'forward':
                entry_prices = {
                    symbols[0]: prices['ask1'],
                    symbols[1]: prices['bid2'],
                    symbols[2]: prices['bid3']
                }
            else:
                entry_prices = {
                    symbols[0]: prices['ask3'],
                    symbols[1]: prices['ask2'],
                    symbols[2]: prices['bid1']
                }
            
            # Calculate sizes
            base_amount = 100  # $100 base amount
            sizes = self._calculate_triangular_sizes_safe(
                base_amount, list(prices.values()), symbols
            )
            
            # Calculate confidence
            confidence = self._calculate_triangular_confidence_safe(market_data, symbols)
            
            return ArbitrageOpportunity(
                type=f'triangular_{direction}',
                symbols=symbols,
                entry_prices=entry_prices,
                sizes=sizes,
                expected_profit=profit * base_amount,
                expected_profit_pct=profit,
                confidence=confidence,
                execution_time_estimate=0.3,
                risk_score=self._calculate_risk_score(profit, 0.3)
            )
            
        except Exception as e:
            logger.error(f"Error creating triangular opportunity: {e}")
            raise
    
    def _calculate_triangular_sizes_safe(self, base_amount: float, prices: List[float],
                                       symbols: List[str]) -> Dict[str, float]:
        """Calculate position sizes with error handling"""
        try:
            sizes = {}
            
            # Validate inputs
            if not prices or len(prices) < 3:
                raise ValueError("Insufficient price data")
            
            # Simple size calculation with validation
            for i, symbol in enumerate(symbols):
                if i == 0:
                    size = base_amount / prices[0] if prices[0] > 0 else 0
                elif i == 1:
                    size = sizes.get(symbols[0], 0)  # Same as first leg
                else:
                    size = base_amount / prices[2] if prices[2] > 0 else 0
                
                # Validate size
                if np.isfinite(size) and size > 0:
                    sizes[symbol] = size
                else:
                    sizes[symbol] = 0
                    
            return sizes
            
        except Exception as e:
            logger.error(f"Error calculating triangular sizes: {e}")
            # Return minimal sizes
            return {symbol: 0.001 for symbol in symbols}
    
    def _calculate_triangular_confidence_safe(self, market_data: Dict, symbols: List[str]) -> float:
        """Calculate confidence with error handling"""
        try:
            confidence_factors = []
            
            for symbol in symbols:
                if symbol not in market_data:
                    continue
                    
                data = market_data[symbol]
                
                # Check spread tightness
                if 'spread' in data and 'mid_price' in data:
                    spread = data['spread']
                    mid_price = data['mid_price']
                    
                    if mid_price > 0:
                        spread_pct = spread / mid_price
                        # Tighter spread = higher confidence
                        spread_confidence = max(0, 1 - spread_pct * 100)
                        confidence_factors.append(spread_confidence)
                
                # Check liquidity
                if 'bid_volume' in data and 'ask_volume' in data:
                    volume = data['bid_volume'] + data['ask_volume']
                    liquidity_confidence = min(1, volume / 10000)  # Normalize by $10k
                    confidence_factors.append(liquidity_confidence)
            
            if confidence_factors:
                return np.mean(confidence_factors)
            else:
                return 0.5  # Default confidence
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_risk_score(self, profit: float, execution_time: float) -> float:
        """Calculate risk score for opportunity"""
        # Higher profit = lower risk
        profit_factor = min(profit / 0.01, 1.0)  # Normalize by 1% profit
        
        # Faster execution = lower risk
        time_factor = max(0, 1 - execution_time / 2.0)  # Normalize by 2 seconds
        
        # Combined risk score (lower is better)
        risk_score = 1 - (profit_factor * 0.7 + time_factor * 0.3)
        
        return risk_score
    
    async def execute_triangular_arbitrage(self, opportunity: ArbitrageOpportunity, 
                                         executor, risk_manager) -> Dict:
        """Execute triangular arbitrage with error handling"""
        orders = []
        executed_orders = []
        
        try:
            # Validate executor
            if not executor:
                raise ValueError("No executor available")
            
            # Check risk limits
            total_required = sum(
                abs(size) * opportunity.entry_prices.get(symbol, 0)
                for symbol, size in opportunity.sizes.items()
            )
            
            available_capital = risk_manager.get_available_capital()
            if total_required > available_capital:
                logger.warning(f"Insufficient capital: required {total_required}, available {available_capital}")
                return {
                    'status': 'rejected',
                    'error': 'Insufficient capital',
                    'executed_orders': []
                }
            
            # Prepare all orders with validation
            for symbol in opportunity.symbols:
                size = opportunity.sizes.get(symbol, 0)
                price = opportunity.entry_prices.get(symbol, 0)
                
                # Validate order parameters
                if size <= 0 or price <= 0:
                    raise ValueError(f"Invalid order parameters for {symbol}: size={size}, price={price}")
                
                # Determine side based on triangular direction
                if opportunity.type == 'triangular_forward':
                    side = 'buy' if symbol == opportunity.symbols[0] else 'sell'
                else:  # reverse
                    side = 'sell' if symbol == opportunity.symbols[0] else 'buy'
                
                orders.append({
                    'symbol': symbol,
                    'side': side,
                    'size': abs(size),
                    'price': price,
                    'type': 'limit',
                    'time_in_force': 'IOC'  # Immediate or cancel
                })
            
            # Execute with timeout
            timeout = self.params['max_execution_time']
            
            # Execute all orders simultaneously with timeout
            tasks = []
            for order in orders:
                task = asyncio.create_task(
                    self._place_order_with_timeout(executor, order, timeout)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            all_success = True
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Order {i} failed: {result}")
                    executed_orders.append({'status': 'failed', 'error': str(result)})
                    all_success = False
                elif isinstance(result, dict) and result.get('status') == 'filled':
                    executed_orders.append(result)
                else:
                    executed_orders.append({'status': 'failed', 'result': result})
                    all_success = False
            
            if all_success:
                # Calculate actual profit
                actual_profit = self._calculate_actual_profit(executed_orders, orders)
                
                return {
                    'status': 'success',
                    'executed_orders': executed_orders,
                    'profit_realized': actual_profit
                }
            else:
                # Attempt to unwind any filled orders
                await self._unwind_partial_arbitrage(executed_orders, orders, executor)
                
                return {
                    'status': 'partial_fill',
                    'executed_orders': executed_orders,
                    'profit_realized': 0
                }
                
        except Exception as e:
            logger.error(f"Error in triangular arbitrage execution: {e}")
            return {
                'status': 'failed',
                'executed_orders': executed_orders,
                'profit_realized': 0,
                'error': str(e)
            }
    
    async def _place_order_with_timeout(self, executor, order: Dict, timeout: float) -> Dict:
        """Place order with timeout"""
        try:
            return await asyncio.wait_for(
                executor.place_order_async(**order),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Order timeout for {order['symbol']}")
            raise
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            raise
    
    def _calculate_actual_profit(self, results: List[Dict], orders: List[Dict]) -> float:
        """Calculate actual profit with error handling"""
        try:
            total_cost = 0
            total_revenue = 0
            
            for result, order in zip(results, orders):
                if not isinstance(result, dict) or result.get('status') != 'filled':
                    continue
                    
                fill_price = result.get('fill_price', order['price'])
                size = result.get('size', order['size'])
                fee = result.get('fee', 0)
                
                # Validate values
                if not all(np.isfinite(x) for x in [fill_price, size, fee]):
                    logger.warning(f"Invalid fill values: price={fill_price}, size={size}, fee={fee}")
                    continue
                
                if order['side'] == 'buy':
                    total_cost += fill_price * size + fee
                else:
                    total_revenue += fill_price * size - fee
            
            profit = total_revenue - total_cost
            
            # Sanity check
            if abs(profit) > total_cost * 0.5:  # More than 50% profit/loss is suspicious
                logger.warning(f"Suspicious profit calculation: {profit} (cost: {total_cost})")
                return 0
                
            return profit
            
        except Exception as e:
            logger.error(f"Error calculating triangular profit: {e}")
            return 0
    
    async def _unwind_partial_arbitrage(self, results: List, orders: List[Dict], executor):
        """Safely unwind partially filled arbitrage orders"""
        try:
            unwind_tasks = []
            
            for result, order in zip(results, orders):
                if isinstance(result, dict) and result.get('status') == 'filled':
                    # Create opposite order to unwind
                    unwind_order = {
                        'symbol': order['symbol'],
                        'side': 'sell' if order['side'] == 'buy' else 'buy',
                        'size': result.get('size', order['size']),
                        'type': 'market'  # Use market order for immediate execution
                    }
                    
                    task = asyncio.create_task(
                        self._place_order_with_timeout(executor, unwind_order, 5.0)
                    )
                    unwind_tasks.append(task)
            
            if unwind_tasks:
                # Execute unwind orders with error handling
                unwind_results = await asyncio.gather(*unwind_tasks, return_exceptions=True)
                
                success_count = sum(
                    1 for r in unwind_results 
                    if isinstance(r, dict) and r.get('status') == 'filled'
                )
                
                logger.info(f"Unwound {success_count}/{len(unwind_tasks)} positions")
                
        except Exception as e:
            logger.error(f"Error unwinding partial arbitrage: {e}")
