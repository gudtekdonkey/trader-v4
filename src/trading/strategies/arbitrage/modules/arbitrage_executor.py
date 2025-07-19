"""Arbitrage execution manager and coordinator"""

import asyncio
import pandas as pd
from typing import Dict, List, Optional
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


class ArbitrageExecutor:
    """Manages arbitrage opportunity execution and lifecycle"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.active_arbitrages = {}
        self.execution_history = []
        self.performance = {
            'total_opportunities': 0,
            'executed_opportunities': 0,
            'successful_arbitrages': 0,
            'total_profit': 0,
            'failed_arbitrages': 0,
            'errors': 0
        }
    
    async def execute_opportunity(self, opportunity, executor, risk_manager, 
                                 triangular_handler, statistical_handler, 
                                 funding_handler) -> Dict:
        """Execute arbitrage opportunity with comprehensive error handling"""
        execution_result = {
            'opportunity': opportunity,
            'status': 'pending',
            'executed_orders': [],
            'profit_realized': 0,
            'execution_time': 0,
            'retry_count': 0
        }
        
        # Check risk limits
        if not self._check_execution_risk_limits(opportunity, risk_manager):
            execution_result['status'] = 'rejected'
            execution_result['error'] = 'Risk limits exceeded'
            return execution_result
        
        start_time = asyncio.get_event_loop().time()
        
        # Retry mechanism
        for attempt in range(self.params['max_retry_attempts']):
            try:
                execution_result['retry_count'] = attempt
                
                # Execute based on type
                if opportunity.type.startswith('triangular'):
                    result = await triangular_handler.execute_triangular_arbitrage(
                        opportunity, executor, risk_manager
                    )
                elif opportunity.type == 'statistical':
                    result = await statistical_handler.execute_statistical_arbitrage(
                        opportunity, executor, risk_manager
                    )
                elif opportunity.type == 'funding':
                    result = await funding_handler.execute_funding_arbitrage(
                        opportunity, executor, risk_manager
                    )
                else:
                    raise ValueError(f"Unknown arbitrage type: {opportunity.type}")
                
                execution_result.update(result)
                execution_result['execution_time'] = asyncio.get_event_loop().time() - start_time
                
                # Update performance tracking
                if result['status'] == 'success':
                    self.performance['successful_arbitrages'] += 1
                    self.performance['total_profit'] += result.get('profit_realized', 0)
                    
                    # Track active arbitrage
                    self._track_active_arbitrage(opportunity, result)
                    break  # Success, exit retry loop
                else:
                    if attempt < self.params['max_retry_attempts'] - 1:
                        await asyncio.sleep(self.params['retry_delay'])
                        continue
                    else:
                        self.performance['failed_arbitrages'] += 1
                
                self.performance['executed_opportunities'] += 1
                
            except Exception as e:
                logger.error(f"Failed to execute arbitrage (attempt {attempt + 1}): {e}")
                execution_result['status'] = 'failed'
                execution_result['error'] = str(e)
                
                if attempt < self.params['max_retry_attempts'] - 1:
                    await asyncio.sleep(self.params['retry_delay'])
                else:
                    self.performance['failed_arbitrages'] += 1
                    self.performance['errors'] += 1
        
        # Record execution history
        self.execution_history.append(execution_result)
        
        return execution_result
    
    def _check_execution_risk_limits(self, opportunity, risk_manager) -> bool:
        """Check if execution passes risk limits"""
        try:
            # Calculate total required capital
            total_required = sum(
                abs(size) * opportunity.entry_prices.get(symbol, 0)
                for symbol, size in opportunity.sizes.items()
            )
            
            available_capital = risk_manager.get_available_capital()
            if total_required > available_capital:
                logger.warning(f"Insufficient capital: required {total_required}, available {available_capital}")
                return False
            
            # Check position limits for each symbol
            for symbol in opportunity.symbols:
                if not risk_manager.check_pre_trade_risk(
                    symbol, 'buy', 0.001, 1.0  # Minimal check
                )[0]:
                    logger.warning(f"Risk check failed for {symbol}")
                    return False
            
            # Check maximum arbitrage exposure
            current_exposure = self._calculate_current_exposure()
            max_exposure = self.params.get('max_arbitrage_exposure', 0.2) * available_capital
            
            if current_exposure + total_required > max_exposure:
                logger.warning(f"Would exceed max arbitrage exposure: {current_exposure + total_required} > {max_exposure}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def _calculate_current_exposure(self) -> float:
        """Calculate current total arbitrage exposure"""
        total_exposure = 0
        
        for arb_id, arb_data in self.active_arbitrages.items():
            for symbol, position in arb_data.get('positions', {}).items():
                exposure = abs(position['size']) * position.get('current_price', position['entry_price'])
                total_exposure += exposure
        
        return total_exposure
    
    def _track_active_arbitrage(self, opportunity, execution_result):
        """Track active arbitrage position"""
        arb_id = f"{opportunity.type}_{pd.Timestamp.now().timestamp()}"
        
        positions = {}
        for order in execution_result.get('executed_orders', []):
            if order.get('status') == 'filled':
                symbol = order['symbol']
                positions[symbol] = {
                    'side': order['side'],
                    'size': order['size'],
                    'entry_price': order.get('fill_price', order['price']),
                    'current_price': order.get('fill_price', order['price']),
                    'unrealized_pnl': 0
                }
        
        self.active_arbitrages[arb_id] = {
            'type': opportunity.type,
            'symbols': opportunity.symbols,
            'positions': positions,
            'entry_time': pd.Timestamp.now(),
            'expected_profit': opportunity.expected_profit,
            'realized_profit': execution_result.get('profit_realized', 0),
            'status': 'active'
        }
    
    def update_active_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Update active arbitrage positions with current prices"""
        actions = []
        
        for arb_id, arb_data in list(self.active_arbitrages.items()):
            if arb_data['status'] != 'active':
                continue
            
            # Update prices and calculate PnL
            total_pnl = 0
            for symbol, position in arb_data['positions'].items():
                if symbol in current_prices:
                    position['current_price'] = current_prices[symbol]
                    
                    # Calculate unrealized PnL
                    if position['side'] == 'buy':
                        pnl = (position['current_price'] - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - position['current_price']) * position['size']
                    
                    position['unrealized_pnl'] = pnl
                    total_pnl += pnl
            
            arb_data['unrealized_pnl'] = total_pnl
            
            # Check if arbitrage should be closed
            if self._should_close_arbitrage(arb_id, arb_data):
                actions.append({
                    'action': 'close_arbitrage',
                    'arb_id': arb_id,
                    'reason': 'exit_conditions_met',
                    'unrealized_pnl': total_pnl
                })
        
        return actions
    
    def _should_close_arbitrage(self, arb_id: str, arb_data: Dict) -> bool:
        """Determine if arbitrage position should be closed"""
        try:
            # Close if profit target reached
            if arb_data['unrealized_pnl'] >= arb_data['expected_profit'] * 0.8:
                return True
            
            # Close if stop loss hit
            if arb_data['unrealized_pnl'] <= -arb_data['expected_profit'] * 0.5:
                return True
            
            # Close if position too old (depends on type)
            age = pd.Timestamp.now() - arb_data['entry_time']
            
            if arb_data['type'].startswith('triangular') and age.seconds > 60:
                return True  # Triangular should execute quickly
            
            if arb_data['type'] == 'statistical' and age.days > 5:
                return True  # Stat arb can run longer
            
            if arb_data['type'] == 'funding' and age.days > 8:
                return True  # Funding positions reset every 8 hours
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking close conditions: {e}")
            return False
    
    async def close_arbitrage_position(self, arb_id: str, executor) -> Dict:
        """Close an active arbitrage position"""
        try:
            if arb_id not in self.active_arbitrages:
                return {'status': 'not_found'}
            
            arb_data = self.active_arbitrages[arb_id]
            close_orders = []
            
            # Create closing orders
            for symbol, position in arb_data['positions'].items():
                close_order = {
                    'symbol': symbol,
                    'side': 'sell' if position['side'] == 'buy' else 'buy',
                    'size': position['size'],
                    'type': 'market'  # Use market orders for closing
                }
                close_orders.append(close_order)
            
            # Execute closing orders
            tasks = []
            for order in close_orders:
                task = asyncio.create_task(executor.place_order_async(**order))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate realized profit
            total_profit = 0
            for result, order in zip(results, close_orders):
                if isinstance(result, dict) and result.get('status') == 'filled':
                    symbol = order['symbol']
                    position = arb_data['positions'][symbol]
                    
                    if position['side'] == 'buy':
                        profit = (result['fill_price'] - position['entry_price']) * position['size']
                    else:
                        profit = (position['entry_price'] - result['fill_price']) * position['size']
                    
                    total_profit += profit
            
            # Update arbitrage status
            arb_data['status'] = 'closed'
            arb_data['close_time'] = pd.Timestamp.now()
            arb_data['realized_profit'] = total_profit
            
            # Update performance
            self.performance['total_profit'] += total_profit
            
            return {
                'status': 'closed',
                'arb_id': arb_id,
                'realized_profit': total_profit,
                'close_orders': results
            }
            
        except Exception as e:
            logger.error(f"Error closing arbitrage position: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """Get arbitrage performance metrics"""
        return {
            'performance': self.performance.copy(),
            'active_positions': len([a for a in self.active_arbitrages.values() if a['status'] == 'active']),
            'total_positions': len(self.active_arbitrages),
            'recent_executions': self.execution_history[-10:] if self.execution_history else []
        }
