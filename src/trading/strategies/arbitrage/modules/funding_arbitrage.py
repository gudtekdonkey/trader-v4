"""Funding rate arbitrage opportunity detection and execution"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class FundingArbitrage:
    """Handles funding rate arbitrage detection and execution"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.active_funding_positions = {}
    
    def find_funding_opportunities(self, market_data: Dict[str, Dict], 
                                 funding_rates: Dict[str, float]) -> List:
        """Find funding rate arbitrage opportunities"""
        opportunities = []
        
        try:
            if not funding_rates:
                return []
            
            for symbol, rate in funding_rates.items():
                if symbol not in market_data:
                    continue
                
                # Check if funding rate exceeds threshold
                if abs(rate) >= self.params['funding_rate_threshold']:
                    opp = self._create_funding_opportunity(
                        symbol, rate, market_data[symbol]
                    )
                    if opp:
                        opportunities.append(opp)
            
            # Check cross-exchange funding arbitrage
            # This would compare rates across different exchanges
            cross_exchange_opps = self._find_cross_exchange_funding(
                market_data, funding_rates
            )
            opportunities.extend(cross_exchange_opps)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in find_funding_opportunities: {e}")
            return []
    
    def _create_funding_opportunity(self, symbol: str, funding_rate: float, 
                                  market_data: Dict) -> Optional:
        """Create funding rate arbitrage opportunity"""
        try:
            from .triangular_arbitrage import ArbitrageOpportunity
            
            # Determine direction based on funding rate
            if funding_rate > 0:  # Longs pay shorts
                # We want to be short to collect funding
                direction = 'short'
                entry_price = market_data['best_bid']
            else:  # Shorts pay longs
                # We want to be long to collect funding
                direction = 'long'
                entry_price = market_data['best_ask']
            
            # Calculate expected profit (8-hour funding period)
            hours_per_period = 8
            periods_per_day = 24 / hours_per_period
            
            # Daily profit from funding
            daily_funding_profit = abs(funding_rate) * periods_per_day
            
            # Account for entry/exit costs
            spread_cost = (market_data['best_ask'] - market_data['best_bid']) / market_data['mid_price']
            fee_cost = 0.001 * 2  # Entry and exit fees
            
            # Net expected profit
            expected_profit_pct = daily_funding_profit - spread_cost - fee_cost
            
            if expected_profit_pct <= 0:
                return None
            
            # Calculate position size
            base_amount = min(
                500,  # $500 for funding arb
                self.params.get('max_position_size', 0.1) * 5000
            )
            
            size = base_amount / entry_price
            
            # Calculate confidence
            # Higher funding rate = higher confidence
            confidence = min(abs(funding_rate) / 0.01, 1.0) * 0.8
            
            return ArbitrageOpportunity(
                type='funding',
                symbols=[symbol],
                entry_prices={symbol: entry_price},
                sizes={symbol: size},
                expected_profit=expected_profit_pct * base_amount,
                expected_profit_pct=expected_profit_pct,
                confidence=confidence,
                execution_time_estimate=1.0,
                risk_score=1 - confidence
            )
            
        except Exception as e:
            logger.error(f"Error creating funding opportunity: {e}")
            return None
    
    def _find_cross_exchange_funding(self, market_data: Dict[str, Dict], 
                                   funding_rates: Dict[str, float]) -> List:
        """Find cross-exchange funding arbitrage opportunities"""
        opportunities = []
        
        try:
            # In practice, this would compare funding rates across exchanges
            # For now, we'll simulate with synthetic data
            
            # Example: BTC perpetual vs spot
            if 'BTC-USD' in market_data and 'BTC-PERP' in funding_rates:
                perp_rate = funding_rates['BTC-PERP']
                
                if abs(perp_rate) > self.params['funding_rate_threshold']:
                    # Create synthetic arbitrage between spot and perp
                    spot_data = market_data['BTC-USD']
                    
                    from .triangular_arbitrage import ArbitrageOpportunity
                    
                    # Calculate basis
                    basis = 0.0  # Would calculate from perp vs spot prices
                    
                    # Expected profit from funding + basis convergence
                    expected_profit_pct = abs(perp_rate) * 3 + basis * 0.5
                    
                    if expected_profit_pct > self.params['min_profit_threshold']:
                        opp = ArbitrageOpportunity(
                            type='funding_basis',
                            symbols=['BTC-USD', 'BTC-PERP'],
                            entry_prices={
                                'BTC-USD': spot_data['best_ask'],
                                'BTC-PERP': spot_data['best_bid']  # Synthetic
                            },
                            sizes={
                                'BTC-USD': 0.01,
                                'BTC-PERP': 0.01
                            },
                            expected_profit=expected_profit_pct * 1000,
                            expected_profit_pct=expected_profit_pct,
                            confidence=0.7,
                            execution_time_estimate=2.0,
                            risk_score=0.3
                        )
                        opportunities.append(opp)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in cross-exchange funding search: {e}")
            return []
    
    async def execute_funding_arbitrage(self, opportunity, executor, risk_manager) -> Dict:
        """Execute funding rate arbitrage"""
        try:
            symbol = opportunity.symbols[0]
            
            # Determine order parameters
            if opportunity.entry_prices[symbol] == executor.get_market_data(symbol)['best_bid']:
                side = 'sell'  # Short position
            else:
                side = 'buy'   # Long position
            
            # Place order
            order = {
                'symbol': symbol,
                'side': side,
                'size': opportunity.sizes[symbol],
                'price': opportunity.entry_prices[symbol],
                'type': 'limit',
                'time_in_force': 'GTC'
            }
            
            result = await executor.place_order_async(**order)
            
            if result.get('status') == 'success':
                # Track funding position
                self.active_funding_positions[symbol] = {
                    'side': side,
                    'size': opportunity.sizes[symbol],
                    'entry_price': opportunity.entry_prices[symbol],
                    'entry_time': pd.Timestamp.now(),
                    'expected_funding': opportunity.expected_profit_pct
                }
                
                return {
                    'status': 'success',
                    'executed_orders': [result],
                    'profit_realized': 0  # Will be realized at funding time
                }
            else:
                return {
                    'status': 'failed',
                    'executed_orders': [result],
                    'profit_realized': 0
                }
                
        except Exception as e:
            logger.error(f"Error executing funding arbitrage: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'executed_orders': []
            }
    
    def collect_funding_payments(self, symbol: str, funding_payment: float) -> Dict:
        """Process funding payment collection"""
        try:
            if symbol not in self.active_funding_positions:
                return {'status': 'no_position'}
            
            position = self.active_funding_positions[symbol]
            
            # Calculate actual vs expected
            expected = position['expected_funding'] * position['size'] * position['entry_price']
            actual = funding_payment
            
            logger.info(f"Collected funding for {symbol}: ${actual:.2f} (expected: ${expected:.2f})")
            
            return {
                'status': 'collected',
                'symbol': symbol,
                'amount': actual,
                'expected': expected,
                'efficiency': actual / expected if expected != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error collecting funding: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def should_close_funding_position(self, symbol: str, current_funding_rate: float) -> bool:
        """Determine if funding position should be closed"""
        try:
            if symbol not in self.active_funding_positions:
                return False
            
            position = self.active_funding_positions[symbol]
            
            # Close if funding rate has flipped or decreased significantly
            if position['side'] == 'sell' and current_funding_rate < 0:
                return True  # Was collecting positive funding, now negative
            
            if position['side'] == 'buy' and current_funding_rate > 0:
                return True  # Was collecting negative funding, now positive
            
            # Close if rate has decreased by more than 50%
            if abs(current_funding_rate) < abs(position['expected_funding']) * 0.5:
                return True
            
            # Close if position has been open for more than 7 days
            if (pd.Timestamp.now() - position['entry_time']).days > 7:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking funding close conditions: {e}")
            return False
