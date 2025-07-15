import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from ..risk_manager import RiskManager
from ...utils.logger import setup_logger

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

class ArbitrageStrategy:
    """Multi-type arbitrage strategy for Hyperliquid"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.active_arbitrages = {}
        
        # Strategy parameters
        self.params = {
            'min_profit_threshold': 0.002,  # 0.2% minimum profit
            'max_execution_time': 1.0,  # 1 second max execution
            'funding_rate_threshold': 0.001,  # 0.1% funding rate difference
            'correlation_threshold': 0.8,  # For statistical arbitrage
            'z_score_entry': 2.0,
            'z_score_exit': 0.5,
            'max_position_size': 0.1,  # 10% of capital per arbitrage
            'slippage_buffer': 0.0005,  # 0.05% slippage allowance
            'min_confidence': 0.7
        }
        
        # Performance tracking
        self.performance = {
            'total_opportunities': 0,
            'executed_opportunities': 0,
            'successful_arbitrages': 0,
            'total_profit': 0,
            'failed_arbitrages': 0
        }
        
        # Statistical arbitrage pairs
        self.stat_arb_pairs = []
        self.pair_models = {}
        
    def find_opportunities(self, market_data: Dict[str, Dict], 
                         funding_rates: Optional[Dict[str, float]] = None) -> List[ArbitrageOpportunity]:
        """Find all types of arbitrage opportunities"""
        opportunities = []
        
        # Triangular arbitrage
        triangular_opps = self._find_triangular_arbitrage(market_data)
        opportunities.extend(triangular_opps)
        
        # Statistical arbitrage
        stat_arb_opps = self._find_statistical_arbitrage(market_data)
        opportunities.extend(stat_arb_opps)
        
        # Funding rate arbitrage
        if funding_rates:
            funding_opps = self._find_funding_arbitrage(market_data, funding_rates)
            opportunities.extend(funding_opps)
        
        # Filter by confidence and profitability
        filtered_opportunities = [
            opp for opp in opportunities
            if opp.confidence >= self.params['min_confidence'] and
               opp.expected_profit_pct >= self.params['min_profit_threshold']
        ]
        
        # Sort by expected profit
        filtered_opportunities.sort(key=lambda x: x.expected_profit_pct, reverse=True)
        
        return filtered_opportunities
    
    def _find_triangular_arbitrage(self, market_data: Dict[str, Dict]) -> List[ArbitrageOpportunity]:
        """Find triangular arbitrage opportunities"""
        opportunities = []
        
        # Get all available trading pairs
        symbols = list(market_data.keys())
        
        # Find triangular paths (e.g., BTC/USDT -> ETH/BTC -> ETH/USDT)
        for i, symbol1 in enumerate(symbols):
            base1, quote1 = self._parse_symbol(symbol1)
            
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    continue
                    
                base2, quote2 = self._parse_symbol(symbol2)
                
                # Check if we can form a triangle
                if quote1 == base2:  # e.g., BTC/USDT and ETH/BTC
                    # Look for closing pair
                    closing_symbol = f"{base1}/{quote2}"
                    if closing_symbol in symbols:
                        # Calculate triangular arbitrage
                        opp = self._calculate_triangular_arbitrage(
                            symbol1, symbol2, closing_symbol, market_data
                        )
                        
                        if opp and opp.expected_profit_pct > self.params['min_profit_threshold']:
                            opportunities.append(opp)
        
        return opportunities
    
    def _calculate_triangular_arbitrage(self, symbol1: str, symbol2: str, symbol3: str,
                                      market_data: Dict[str, Dict]) -> Optional[ArbitrageOpportunity]:
        """Calculate profitability of triangular arbitrage"""
        try:
            # Get best bid/ask for each pair
            bid1 = market_data[symbol1]['best_bid']
            ask1 = market_data[symbol1]['best_ask']
            bid2 = market_data[symbol2]['best_bid']
            ask2 = market_data[symbol2]['best_ask']
            bid3 = market_data[symbol3]['best_bid']
            ask3 = market_data[symbol3]['best_ask']
            
            # Calculate forward arbitrage (buy symbol1, sell symbol2, sell symbol3)
            # Start with 1 unit of quote currency
            forward_path = 1 / ask1  # Buy base1 with quote1
            forward_path = forward_path * bid2  # Sell base1 for quote2
            forward_path = forward_path * bid3  # Sell base2 for quote1
            
            # Calculate reverse arbitrage
            reverse_path = 1 / ask3  # Buy base2 with quote1
            reverse_path = reverse_path / ask2  # Buy base1 with base2
            reverse_path = reverse_path * bid1  # Sell base1 for quote1
            
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
                return ArbitrageOpportunity(
                    type='triangular_forward',
                    symbols=[symbol1, symbol2, symbol3],
                    entry_prices={
                        symbol1: ask1,
                        symbol2: bid2,
                        symbol3: bid3
                    },
                    sizes=self._calculate_triangular_sizes(100, ask1, bid2, bid3),  # $100 base
                    expected_profit=forward_profit_net * 100,
                    expected_profit_pct=forward_profit_net,
                    confidence=self._calculate_triangular_confidence(market_data, [symbol1, symbol2, symbol3]),
                    execution_time_estimate=0.3,  # 300ms estimate
                    risk_score=self._calculate_risk_score(forward_profit_net, 0.3)
                )
            
            elif reverse_profit_net > 0:
                return ArbitrageOpportunity(
                    type='triangular_reverse',
                    symbols=[symbol3, symbol2, symbol1],
                    entry_prices={
                        symbol3: ask3,
                        symbol2: ask2,
                        symbol1: bid1
                    },
                    sizes=self._calculate_triangular_sizes(100, ask3, 1/ask2, bid1),
                    expected_profit=reverse_profit_net * 100,
                    expected_profit_pct=reverse_profit_net,
                    confidence=self._calculate_triangular_confidence(market_data, [symbol3, symbol2, symbol1]),
                    execution_time_estimate=0.3,
                    risk_score=self._calculate_risk_score(reverse_profit_net, 0.3)
                )
            
        except Exception as e:
            logger.error(f"Error calculating triangular arbitrage: {e}")
        
        return None
    
    def _find_statistical_arbitrage(self, market_data: Dict[str, Dict]) -> List[ArbitrageOpportunity]:
        """Find statistical arbitrage opportunities between correlated pairs"""
        opportunities = []
        
        # Check pre-identified statistical arbitrage pairs
        for pair_info in self.stat_arb_pairs:
            symbol1 = pair_info['symbol1']
            symbol2 = pair_info['symbol2']
            
            if symbol1 in market_data and symbol2 in market_data:
                # Calculate spread z-score
                price1 = market_data[symbol1]['mid_price']
                price2 = market_data[symbol2]['mid_price']
                
                spread = price1 - pair_info['hedge_ratio'] * price2
                z_score = (spread - pair_info['spread_mean']) / pair_info['spread_std']
                
                # Check for entry signals
                if abs(z_score) >= self.params['z_score_entry']:
                    opp = self._create_stat_arb_opportunity(
                        symbol1, symbol2, price1, price2, z_score, pair_info
                    )
                    if opp:
                        opportunities.append(opp)
        
        return opportunities
    
    def _create_stat_arb_opportunity(self, symbol1: str, symbol2: str, price1: float, price2: float,
                                   z_score: float, pair_info: Dict) -> Optional[ArbitrageOpportunity]:
        """Create statistical arbitrage opportunity"""
        # Determine direction
        if z_score > self.params['z_score_entry']:
            # Spread too high: short symbol1, long symbol2
            direction = -1
        else:
            # Spread too low: long symbol1, short symbol2
            direction = 1
        
        # Calculate position sizes
        capital_allocation = 10000  # $10k per stat arb
        size1 = capital_allocation / (2 * price1)
        size2 = pair_info['hedge_ratio'] * capital_allocation / (2 * price2)
        
        # Estimate profit
        spread_reversion = abs(z_score - self.params['z_score_exit']) * pair_info['spread_std']
        expected_profit = spread_reversion * size1
        expected_profit_pct = expected_profit / capital_allocation
        
        # Account for costs
        fee_cost = 0.001 * 4  # Entry and exit for both legs
        expected_profit_pct -= fee_cost
        
        if expected_profit_pct > self.params['min_profit_threshold']:
            return ArbitrageOpportunity(
                type='statistical',
                symbols=[symbol1, symbol2],
                entry_prices={
                    symbol1: price1,
                    symbol2: price2
                },
                sizes={
                    symbol1: size1 * direction,
                    symbol2: -size2 * direction
                },
                expected_profit=expected_profit,
                expected_profit_pct=expected_profit_pct,
                confidence=min(0.9, pair_info['correlation']),
                execution_time_estimate=0.2,
                risk_score=self._calculate_risk_score(expected_profit_pct, 0.2)
            )
        
        return None
    
    def _find_funding_arbitrage(self, market_data: Dict[str, Dict], 
                              funding_rates: Dict[str, float]) -> List[ArbitrageOpportunity]:
        """Find funding rate arbitrage opportunities"""
        opportunities = []
        
        for symbol, funding_rate in funding_rates.items():
            if symbol in market_data:
                # Check if funding rate is significant
                if abs(funding_rate) > self.params['funding_rate_threshold']:
                    spot_symbol = self._get_spot_symbol(symbol)
                    
                    if spot_symbol and spot_symbol in market_data:
                        opp = self._create_funding_arb_opportunity(
                            symbol, spot_symbol, funding_rate, market_data
                        )
                        if opp:
                            opportunities.append(opp)
        
        return opportunities
    
    def _create_funding_arb_opportunity(self, perp_symbol: str, spot_symbol: str,
                                      funding_rate: float, market_data: Dict[str, Dict]) -> Optional[ArbitrageOpportunity]:
        """Create funding rate arbitrage opportunity"""
        perp_price = market_data[perp_symbol]['mid_price']
        spot_price = market_data[spot_symbol]['mid_price']
        
        # Calculate annualized funding rate
        # Hyperliquid funding is typically 8-hourly
        annualized_rate = funding_rate * 3 * 365
        
        # Only worth it if annualized rate is significant
        if abs(annualized_rate) > 0.1:  # 10% annualized
            # If funding is positive, short perp and long spot
            # If funding is negative, long perp and short spot
            direction = -1 if funding_rate > 0 else 1
            
            capital = 10000
            size = capital / perp_price
            
            # Expected profit from one funding period (8 hours)
            expected_profit = abs(funding_rate) * capital
            expected_profit_pct = abs(funding_rate)
            
            # Account for execution costs
            fee_cost = 0.001 * 2  # Entry fees
            expected_profit_pct -= fee_cost
            
            if expected_profit_pct > self.params['min_profit_threshold']:
                return ArbitrageOpportunity(
                    type='funding',
                    symbols=[perp_symbol, spot_symbol],
                    entry_prices={
                        perp_symbol: perp_price,
                        spot_symbol: spot_price
                    },
                    sizes={
                        perp_symbol: size * direction,
                        spot_symbol: -size * direction
                    },
                    expected_profit=expected_profit,
                    expected_profit_pct=expected_profit_pct,
                    confidence=0.95,  # High confidence for funding arb
                    execution_time_estimate=0.5,
                    risk_score=0.2  # Low risk
                )
        
        return None
    
    async def execute_opportunity(self, opportunity: ArbitrageOpportunity, executor) -> Dict:
        """Execute arbitrage opportunity with atomic-like execution"""
        execution_result = {
            'opportunity': opportunity,
            'status': 'pending',
            'executed_orders': [],
            'profit_realized': 0,
            'execution_time': 0
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute all legs simultaneously
            if opportunity.type.startswith('triangular'):
                result = await self._execute_triangular_arbitrage(opportunity, executor)
            elif opportunity.type == 'statistical':
                result = await self._execute_statistical_arbitrage(opportunity, executor)
            elif opportunity.type == 'funding':
                result = await self._execute_funding_arbitrage(opportunity, executor)
            else:
                raise ValueError(f"Unknown arbitrage type: {opportunity.type}")
            
            execution_result.update(result)
            execution_result['execution_time'] = asyncio.get_event_loop().time() - start_time
            
            # Update performance tracking
            if result['status'] == 'success':
                self.performance['successful_arbitrages'] += 1
                self.performance['total_profit'] += result['profit_realized']
            else:
                self.performance['failed_arbitrages'] += 1
            
            self.performance['executed_opportunities'] += 1
            
        except Exception as e:
            logger.error(f"Failed to execute arbitrage: {e}")
            execution_result['status'] = 'failed'
            execution_result['error'] = str(e)
        
        return execution_result
    
    async def _execute_triangular_arbitrage(self, opportunity: ArbitrageOpportunity, executor) -> Dict:
        """Execute triangular arbitrage atomically"""
        orders = []
        
        # Prepare all orders
        for symbol in opportunity.symbols:
            size = opportunity.sizes[symbol]
            price = opportunity.entry_prices[symbol]
            
            # Determine side based on triangular direction
            if opportunity.type == 'triangular_forward':
                if symbol == opportunity.symbols[0]:
                    side = 'buy'
                else:
                    side = 'sell'
            else:  # reverse
                if symbol == opportunity.symbols[0]:
                    side = 'sell'
                else:
                    side = 'buy'
            
            orders.append({
                'symbol': symbol,
                'side': side,
                'size': abs(size),
                'price': price,
                'type': 'limit',
                'time_in_force': 'IOC'  # Immediate or cancel
            })
        
        # Execute all orders simultaneously
        tasks = [executor.place_order_async(**order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if all orders were successful
        all_success = all(
            isinstance(r, dict) and r.get('status') == 'filled'
            for r in results
        )
        
        if all_success:
            # Calculate actual profit
            actual_profit = self._calculate_actual_triangular_profit(results, orders)
            
            return {
                'status': 'success',
                'executed_orders': results,
                'profit_realized': actual_profit
            }
        else:
            # Attempt to unwind any filled orders
            await self._unwind_partial_arbitrage(results, orders, executor)
            
            return {
                'status': 'partial_fill',
                'executed_orders': results,
                'profit_realized': 0
            }
    
    async def _execute_statistical_arbitrage(self, opportunity: ArbitrageOpportunity, executor) -> Dict:
        """Execute statistical arbitrage"""
        symbol1, symbol2 = opportunity.symbols
        
        # Execute both legs simultaneously
        order1 = executor.place_order_async(
            symbol=symbol1,
            side='buy' if opportunity.sizes[symbol1] > 0 else 'sell',
            size=abs(opportunity.sizes[symbol1]),
            price=opportunity.entry_prices[symbol1],
            type='limit',
            time_in_force='GTC'
        )
        
        order2 = executor.place_order_async(
            symbol=symbol2,
            side='buy' if opportunity.sizes[symbol2] > 0 else 'sell',
            size=abs(opportunity.sizes[symbol2]),
            price=opportunity.entry_prices[symbol2],
            type='limit',
            time_in_force='GTC'
        )
        
        results = await asyncio.gather(order1, order2, return_exceptions=True)
        
        # Track the arbitrage position
        if all(isinstance(r, dict) and r.get('status') == 'filled' for r in results):
            position_id = f"stat_arb_{symbol1}_{symbol2}_{pd.Timestamp.now()}"
            self.active_arbitrages[position_id] = {
                'type': 'statistical',
                'symbols': opportunity.symbols,
                'entry_prices': {
                    symbol1: results[0]['fill_price'],
                    symbol2: results[1]['fill_price']
                },
                'sizes': opportunity.sizes,
                'entry_z_score': self._calculate_current_z_score(symbol1, symbol2),
                'entry_time': pd.Timestamp.now()
            }
            
            return {
                'status': 'success',
                'executed_orders': results,
                'profit_realized': 0  # Realized on exit
            }
        
        return {
            'status': 'failed',
            'executed_orders': results,
            'profit_realized': 0
        }
    
    async def _execute_funding_arbitrage(self, opportunity: ArbitrageOpportunity, executor) -> Dict:
        """Execute funding rate arbitrage"""
        perp_symbol, spot_symbol = opportunity.symbols
        
        # Execute both legs
        perp_order = executor.place_order_async(
            symbol=perp_symbol,
            side='sell' if opportunity.sizes[perp_symbol] < 0 else 'buy',
            size=abs(opportunity.sizes[perp_symbol]),
            price=opportunity.entry_prices[perp_symbol],
            type='limit',
            time_in_force='GTC'
        )
        
        spot_order = executor.place_order_async(
            symbol=spot_symbol,
            side='buy' if opportunity.sizes[spot_symbol] > 0 else 'sell',
            size=abs(opportunity.sizes[spot_symbol]),
            price=opportunity.entry_prices[spot_symbol],
            type='limit',
            time_in_force='GTC'
        )
        
        results = await asyncio.gather(perp_order, spot_order, return_exceptions=True)
        
        if all(isinstance(r, dict) and r.get('status') == 'filled' for r in results):
            position_id = f"funding_arb_{perp_symbol}_{pd.Timestamp.now()}"
            self.active_arbitrages[position_id] = {
                'type': 'funding',
                'symbols': opportunity.symbols,
                'entry_prices': {
                    perp_symbol: results[0]['fill_price'],
                    spot_symbol: results[1]['fill_price']
                },
                'sizes': opportunity.sizes,
                'entry_time': pd.Timestamp.now(),
                'next_funding_time': pd.Timestamp.now() + pd.Timedelta(hours=8)
            }
            
            return {
                'status': 'success',
                'executed_orders': results,
                'profit_realized': 0  # Realized at funding time
            }
        
        return {
            'status': 'failed',
            'executed_orders': results,
            'profit_realized': 0
        }
    
    def _calculate_triangular_sizes(self, base_amount: float, price1: float, 
                                  price2: float, price3: float) -> Dict[str, float]:
        """Calculate position sizes for triangular arbitrage"""
        # This is simplified - in practice would need more complex calculation
        return {
            'symbol1': base_amount / price1,
            'symbol2': base_amount / price1,  # Adjusted for base currency
            'symbol3': base_amount
        }
    
    def _calculate_triangular_confidence(self, market_data: Dict, symbols: List[str]) -> float:
        """Calculate confidence in triangular arbitrage execution"""
        confidence_factors = []
        
        for symbol in symbols:
            # Check spread tightness
            spread = market_data[symbol]['spread']
            mid_price = market_data[symbol]['mid_price']
            spread_pct = spread / mid_price
            
            # Tighter spread = higher confidence
            spread_confidence = max(0, 1 - spread_pct * 100)
            confidence_factors.append(spread_confidence)
            
            # Check liquidity
            if 'bid_volume' in market_data[symbol]:
                volume = market_data[symbol]['bid_volume'] + market_data[symbol]['ask_volume']
                liquidity_confidence = min(1, volume / 10000)  # Normalize by $10k
                confidence_factors.append(liquidity_confidence)
        
        return np.mean(confidence_factors)
    
    def _calculate_risk_score(self, expected_profit: float, execution_time: float) -> float:
        """Calculate risk score for arbitrage opportunity"""
        # Higher profit = lower risk
        profit_score = min(1, expected_profit / 0.01)  # Normalize by 1%
        
        # Faster execution = lower risk
        time_score = max(0, 1 - execution_time / self.params['max_execution_time'])
        
        # Combined risk (inverted so lower is better)
        risk_score = 1 - (profit_score * 0.7 + time_score * 0.3)
        
        return risk_score
    
    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parse symbol into base and quote currencies"""
        # Hyperliquid uses format like "BTC-USD"
        parts = symbol.split('-')
        if len(parts) == 2:
            return parts[0], parts[1]
        return symbol[:3], symbol[3:]  # Fallback
    
    def _get_spot_symbol(self, perp_symbol: str) -> Optional[str]:
        """Get corresponding spot symbol for a perpetual"""
        # This would need actual mapping based on Hyperliquid's conventions
        if perp_symbol.endswith('-PERP'):
            return perp_symbol.replace('-PERP', '-SPOT')
        return None
    
    def _calculate_actual_triangular_profit(self, results: List[Dict], orders: List[Dict]) -> float:
        """Calculate actual profit from executed triangular arbitrage"""
        # Track the flow of funds through the triangular path
        # This is simplified - actual calculation would be more complex
        total_cost = 0
        total_revenue = 0
        
        for result, order in zip(results, orders):
            fill_price = result['fill_price']
            size = result['size']
            fee = result.get('fee', 0)
            
            if order['side'] == 'buy':
                total_cost += fill_price * size + fee
            else:
                total_revenue += fill_price * size - fee
        
        return total_revenue - total_cost
    
    async def _unwind_partial_arbitrage(self, results: List, orders: List[Dict], executor):
        """Unwind any partially filled arbitrage orders"""
        unwind_tasks = []
        
        for result, order in zip(results, orders):
            if isinstance(result, dict) and result.get('status') == 'filled':
                # Create opposite order to unwind
                unwind_order = {
                    'symbol': order['symbol'],
                    'side': 'sell' if order['side'] == 'buy' else 'buy',
                    'size': result['size'],
                    'type': 'market'  # Use market order for immediate execution
                }
                unwind_tasks.append(executor.place_order_async(**unwind_order))
        
        if unwind_tasks:
            await asyncio.gather(*unwind_tasks, return_exceptions=True)
    
    def _calculate_current_z_score(self, symbol1: str, symbol2: str) -> float:
        """Calculate current z-score for a pair"""
        # This would need access to current prices and historical statistics
        # Placeholder implementation
        return 0.0
    
    def update_positions(self, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Check and update arbitrage positions"""
        actions = []
        
        for position_id, position in list(self.active_arbitrages.items()):
            if position['type'] == 'statistical':
                # Check if z-score has reverted
                symbol1, symbol2 = position['symbols']
                if symbol1 in current_data and symbol2 in current_data:
                    current_z_score = self._calculate_current_z_score(symbol1, symbol2)
                    
                    if abs(current_z_score) <= self.params['z_score_exit']:
                        actions.append({
                            'action': 'close_arbitrage',
                            'position_id': position_id,
                            'reason': 'z_score_reversion'
                        })
            
            elif position['type'] == 'funding':
                # Check if funding period has passed
                if pd.Timestamp.now() >= position['next_funding_time']:
                    actions.append({
                        'action': 'collect_funding',
                        'position_id': position_id
                    })
        
        return actions
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return {
            'name': 'Arbitrage',
            'performance': self.performance,
            'active_positions': len(self.active_arbitrages),
            'parameters': self.params,
            'stat_arb_pairs': len(self.stat_arb_pairs)
        }
