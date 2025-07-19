"""Statistical arbitrage opportunity detection and execution"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class StatisticalArbitrage:
    """Handles statistical arbitrage (pairs trading) detection and execution"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.stat_arb_pairs = []
        self.pair_models = {}
    
    def find_statistical_opportunities(self, market_data: Dict[str, Dict]) -> List:
        """Find statistical arbitrage opportunities"""
        opportunities = []
        
        try:
            # Initialize pairs if not done
            if not self.stat_arb_pairs:
                self._initialize_pairs(market_data)
            
            # Check each pair for opportunities
            for pair in self.stat_arb_pairs:
                symbol1, symbol2 = pair['symbol1'], pair['symbol2']
                
                if symbol1 not in market_data or symbol2 not in market_data:
                    continue
                
                # Calculate current z-score
                z_score = self._calculate_z_score(
                    market_data[symbol1]['mid_price'],
                    market_data[symbol2]['mid_price'],
                    pair
                )
                
                # Check for entry signals
                if abs(z_score) >= self.params['z_score_entry']:
                    opp = self._create_stat_arb_opportunity(
                        symbol1, symbol2, z_score, market_data, pair
                    )
                    if opp:
                        opportunities.append(opp)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in find_statistical_opportunities: {e}")
            return []
    
    def _initialize_pairs(self, market_data: Dict[str, Dict]):
        """Initialize statistical arbitrage pairs"""
        try:
            symbols = list(market_data.keys())
            
            # Find correlated pairs
            # In practice, this would use historical data
            # For now, we'll use some predefined pairs
            potential_pairs = [
                ('BTC-USD', 'ETH-USD'),
                ('BTC-USD', 'SOL-USD'),
                ('ETH-USD', 'AVAX-USD'),
                ('MATIC-USD', 'AVAX-USD')
            ]
            
            for symbol1, symbol2 in potential_pairs:
                if symbol1 in symbols and symbol2 in symbols:
                    # Calculate correlation (placeholder)
                    correlation = 0.85  # Would calculate from historical data
                    
                    if correlation >= self.params['correlation_threshold']:
                        self.stat_arb_pairs.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation,
                            'mean_ratio': 1.0,  # Would calculate from history
                            'std_ratio': 0.02,  # Would calculate from history
                            'half_life': 24  # hours
                        })
            
            logger.info(f"Initialized {len(self.stat_arb_pairs)} statistical arbitrage pairs")
            
        except Exception as e:
            logger.error(f"Error initializing pairs: {e}")
    
    def _calculate_z_score(self, price1: float, price2: float, pair: Dict) -> float:
        """Calculate z-score for price ratio"""
        try:
            if price2 == 0:
                return 0
            
            ratio = price1 / price2
            mean_ratio = pair['mean_ratio']
            std_ratio = pair['std_ratio']
            
            if std_ratio == 0:
                return 0
            
            z_score = (ratio - mean_ratio) / std_ratio
            
            return z_score
            
        except Exception as e:
            logger.error(f"Error calculating z-score: {e}")
            return 0
    
    def _create_stat_arb_opportunity(self, symbol1: str, symbol2: str, z_score: float,
                                   market_data: Dict[str, Dict], pair: Dict):
        """Create statistical arbitrage opportunity"""
        try:
            from .triangular_arbitrage import ArbitrageOpportunity
            
            # Determine trade direction
            if z_score > 0:  # Ratio is above mean - short symbol1, long symbol2
                direction = 'short_long'
                entry_prices = {
                    symbol1: market_data[symbol1]['best_bid'],  # Sell symbol1
                    symbol2: market_data[symbol2]['best_ask']   # Buy symbol2
                }
            else:  # Ratio is below mean - long symbol1, short symbol2
                direction = 'long_short'
                entry_prices = {
                    symbol1: market_data[symbol1]['best_ask'],  # Buy symbol1
                    symbol2: market_data[symbol2]['best_bid']   # Sell symbol2
                }
            
            # Calculate expected profit
            mean_reversion_profit = abs(z_score) * pair['std_ratio'] * 0.5  # Conservative estimate
            
            # Account for fees
            fee_cost = 0.001 * 2  # 0.1% fee per trade, 2 trades
            expected_profit_pct = mean_reversion_profit - fee_cost
            
            if expected_profit_pct <= 0:
                return None
            
            # Calculate position sizes (dollar neutral)
            base_amount = min(
                100,  # $100 base
                self.params.get('max_position_size', 0.1) * 1000  # Placeholder capital
            )
            
            sizes = {
                symbol1: base_amount / entry_prices[symbol1],
                symbol2: base_amount / entry_prices[symbol2]
            }
            
            # Calculate confidence based on z-score magnitude
            confidence = min(abs(z_score) / 3.0, 1.0) * pair['correlation']
            
            return ArbitrageOpportunity(
                type='statistical',
                symbols=[symbol1, symbol2],
                entry_prices=entry_prices,
                sizes=sizes,
                expected_profit=expected_profit_pct * base_amount * 2,
                expected_profit_pct=expected_profit_pct,
                confidence=confidence,
                execution_time_estimate=0.5,
                risk_score=1 - confidence
            )
            
        except Exception as e:
            logger.error(f"Error creating stat arb opportunity: {e}")
            return None
    
    async def execute_statistical_arbitrage(self, opportunity, executor, risk_manager) -> Dict:
        """Execute statistical arbitrage trade"""
        try:
            symbol1, symbol2 = opportunity.symbols
            
            # Create orders
            orders = []
            
            # Determine sides based on z-score direction
            if opportunity.type == 'statistical':
                # Check which direction we're trading
                price1 = opportunity.entry_prices[symbol1]
                price2 = opportunity.entry_prices[symbol2]
                
                # If price1 is bid price, we're selling symbol1
                if price1 < price2:  # Rough check
                    orders.append({
                        'symbol': symbol1,
                        'side': 'sell',
                        'size': opportunity.sizes[symbol1],
                        'price': price1,
                        'type': 'limit',
                        'time_in_force': 'GTC'
                    })
                    orders.append({
                        'symbol': symbol2,
                        'side': 'buy',
                        'size': opportunity.sizes[symbol2],
                        'price': price2,
                        'type': 'limit',
                        'time_in_force': 'GTC'
                    })
                else:
                    orders.append({
                        'symbol': symbol1,
                        'side': 'buy',
                        'size': opportunity.sizes[symbol1],
                        'price': price1,
                        'type': 'limit',
                        'time_in_force': 'GTC'
                    })
                    orders.append({
                        'symbol': symbol2,
                        'side': 'sell',
                        'size': opportunity.sizes[symbol2],
                        'price': price2,
                        'type': 'limit',
                        'time_in_force': 'GTC'
                    })
            
            # Execute orders
            tasks = []
            for order in orders:
                task = asyncio.create_task(executor.place_order_async(**order))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            executed_orders = []
            all_success = True
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Order {i} failed: {result}")
                    executed_orders.append({'status': 'failed', 'error': str(result)})
                    all_success = False
                elif isinstance(result, dict) and result.get('status') == 'success':
                    executed_orders.append(result)
                else:
                    executed_orders.append({'status': 'failed', 'result': result})
                    all_success = False
            
            if all_success:
                return {
                    'status': 'success',
                    'executed_orders': executed_orders,
                    'profit_realized': 0  # Will be realized when position closes
                }
            else:
                # Unwind any successful orders
                await self._unwind_stat_arb_orders(executed_orders, orders, executor)
                
                return {
                    'status': 'partial_fill',
                    'executed_orders': executed_orders,
                    'profit_realized': 0
                }
                
        except Exception as e:
            logger.error(f"Error executing statistical arbitrage: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'executed_orders': []
            }
    
    async def _unwind_stat_arb_orders(self, results: List, orders: List[Dict], executor):
        """Unwind partially filled statistical arbitrage orders"""
        try:
            unwind_tasks = []
            
            for result, order in zip(results, orders):
                if isinstance(result, dict) and result.get('status') == 'success':
                    # Create opposite order
                    unwind_order = {
                        'symbol': order['symbol'],
                        'side': 'sell' if order['side'] == 'buy' else 'buy',
                        'size': result.get('size', order['size']),
                        'type': 'market'
                    }
                    
                    task = asyncio.create_task(executor.place_order_async(**unwind_order))
                    unwind_tasks.append(task)
            
            if unwind_tasks:
                await asyncio.gather(*unwind_tasks, return_exceptions=True)
                logger.info(f"Unwound {len(unwind_tasks)} stat arb positions")
                
        except Exception as e:
            logger.error(f"Error unwinding stat arb orders: {e}")
    
    def check_exit_conditions(self, symbol1: str, symbol2: str, 
                            current_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if statistical arbitrage position should be closed"""
        try:
            # Find the pair
            pair = None
            for p in self.stat_arb_pairs:
                if p['symbol1'] == symbol1 and p['symbol2'] == symbol2:
                    pair = p
                    break
            
            if not pair:
                return False
            
            # Calculate current z-score
            if symbol1 in current_data and symbol2 in current_data:
                price1 = current_data[symbol1]['close'].iloc[-1]
                price2 = current_data[symbol2]['close'].iloc[-1]
                
                z_score = self._calculate_z_score(price1, price2, pair)
                
                # Exit if z-score has reverted
                if abs(z_score) <= self.params['z_score_exit']:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
