"""
Exposure Calculator Module

Calculates portfolio exposure metrics for hedge analysis.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ExposureCalculator:
    """
    Calculates various portfolio exposure metrics.
    
    Provides exposure analysis across multiple dimensions including
    market beta, sector concentration, and correlation risks.
    """
    
    def __init__(self):
        """Initialize the exposure calculator."""
        self.default_betas = {
            'BTC': 1.0,
            'ETH': 1.2,
            'SOL': 1.5,
            'AVAX': 1.5,
            'MATIC': 1.5,
            'DOT': 1.5,
            'LINK': 1.3,
            'UNI': 1.4,
            'AAVE': 1.4
        }
        
        self.default_volatilities = {
            'BTC': 0.6,
            'ETH': 0.8,
            'SOL': 1.0,
            'AVAX': 1.0,
            'MATIC': 1.0,
            'DOT': 1.0,
            'LINK': 0.9,
            'UNI': 0.95,
            'AAVE': 0.95,
            'USDT': 0.02,
            'USDC': 0.02,
            'DAI': 0.02
        }
    
    def calculate_exposure_metrics(
        self,
        positions: Dict[str, Dict],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio exposure metrics.
        
        Args:
            positions: Portfolio positions
            market_data: Current market data
            
        Returns:
            Dictionary containing exposure metrics
        """
        try:
            # Calculate total portfolio value
            total_value = self._calculate_total_value(positions)
            
            if total_value <= 0:
                logger.warning("Total portfolio value is zero or negative")
                return {}
            
            # Calculate individual exposures
            exposures = self._calculate_individual_exposures(
                positions, total_value, market_data
            )
            
            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(exposures)
            
            # Calculate concentration metrics
            concentration_metrics = self._calculate_concentration_metrics(exposures)
            
            # Calculate correlation risks
            correlation_risks = self._calculate_correlation_risks(
                exposures, market_data
            )
            
            return {
                'total_value': total_value,
                'individual_exposures': exposures,
                **portfolio_metrics,
                **concentration_metrics,
                **correlation_risks
            }
            
        except Exception as e:
            logger.error(f"Error calculating exposure metrics: {e}")
            return {}
    
    def _calculate_total_value(self, positions: Dict[str, Dict]) -> float:
        """Calculate total portfolio value."""
        total_value = 0
        
        for symbol, pos in positions.items():
            try:
                current_price = pos.get('current_price', pos.get('entry_price', 0))
                size = pos.get('size', 0)
                
                if not isinstance(current_price, (int, float)) or current_price <= 0:
                    logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue
                if not isinstance(size, (int, float)):
                    logger.warning(f"Invalid size for {symbol}: {size}")
                    continue
                
                total_value += size * current_price
                
            except Exception as e:
                logger.error(f"Error calculating value for {symbol}: {e}")
                continue
        
        return total_value
    
    def _calculate_individual_exposures(
        self,
        positions: Dict[str, Dict],
        total_value: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate exposure metrics for each position."""
        exposures = {}
        
        for symbol, position in positions.items():
            try:
                current_price = position.get('current_price', position.get('entry_price', 0))
                size = position.get('size', 0)
                
                if current_price > 0 and isinstance(size, (int, float)):
                    value = size * current_price
                    weight = value / total_value
                    
                    exposures[symbol] = {
                        'weight': weight,
                        'value': value,
                        'size': size,
                        'price': current_price,
                        'beta': self._estimate_beta(symbol, market_data),
                        'volatility': self._estimate_volatility(symbol, market_data),
                        'sector': self._classify_sector(symbol),
                        'correlation_group': self._get_correlation_group(symbol)
                    }
                    
            except Exception as e:
                logger.error(f"Error calculating exposure for {symbol}: {e}")
                continue
        
        return exposures
    
    def _calculate_portfolio_metrics(
        self,
        exposures: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate portfolio-level metrics."""
        if not exposures:
            return {
                'portfolio_beta': 0,
                'portfolio_volatility': 0,
                'portfolio_var': 0
            }
        
        # Portfolio beta (weighted average)
        portfolio_beta = sum(
            exp['weight'] * exp['beta'] 
            for exp in exposures.values()
        )
        
        # Portfolio volatility (simplified - assuming no correlation)
        portfolio_volatility_squared = sum(
            (exp['weight'] * exp['volatility']) ** 2
            for exp in exposures.values()
        )
        portfolio_volatility = np.sqrt(portfolio_volatility_squared)
        
        # Simple VaR calculation (95% confidence, 1-day)
        portfolio_var = 1.645 * portfolio_volatility / np.sqrt(252)
        
        return {
            'portfolio_beta': portfolio_beta,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_var': portfolio_var
        }
    
    def _calculate_concentration_metrics(
        self,
        exposures: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate concentration risk metrics."""
        if not exposures:
            return {
                'max_weight': 0,
                'concentration_index': 0,
                'effective_positions': 0,
                'top3_concentration': 0
            }
        
        weights = [exp['weight'] for exp in exposures.values()]
        
        # Maximum single position weight
        max_weight = max(weights) if weights else 0
        
        # Herfindahl-Hirschman Index
        herfindahl_index = sum(w**2 for w in weights) if weights else 0
        
        # Effective number of positions
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        # Top 3 concentration
        sorted_weights = sorted(weights, reverse=True)
        top3_concentration = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
        
        # Sector concentration
        sector_weights = {}
        for exp in exposures.values():
            sector = exp.get('sector', 'unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + exp['weight']
        
        max_sector_concentration = max(sector_weights.values()) if sector_weights else 0
        
        return {
            'max_weight': max_weight,
            'concentration_index': herfindahl_index,
            'effective_positions': effective_positions,
            'top3_concentration': top3_concentration,
            'max_sector_concentration': max_sector_concentration,
            'sector_weights': sector_weights
        }
    
    def _calculate_correlation_risks(
        self,
        exposures: Dict[str, Dict[str, float]],
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate correlation-based risk metrics."""
        if len(exposures) < 2:
            return {
                'avg_pairwise_correlation': 0,
                'max_correlation': 0,
                'correlation_risk_score': 0
            }
        
        # Calculate average pairwise correlation
        correlations = []
        symbols = list(exposures.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                corr = self._estimate_correlation(symbol1, symbol2, market_data)
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0
        max_correlation = max(correlations) if correlations else 0
        
        # Correlation risk score (weighted by position sizes)
        correlation_risk = 0
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                weight1 = exposures[symbol1]['weight']
                weight2 = exposures[symbol2]['weight']
                corr = self._estimate_correlation(symbol1, symbol2, market_data)
                correlation_risk += weight1 * weight2 * corr
        
        return {
            'avg_pairwise_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'correlation_risk_score': correlation_risk
        }
    
    def _estimate_beta(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Estimate asset beta relative to market."""
        try:
            # Check if beta is available in market data
            if market_data and symbol in market_data:
                symbol_data = market_data.get(symbol, {})
                if isinstance(symbol_data, dict) and 'beta' in symbol_data:
                    return symbol_data['beta']
            
            # Use default betas
            symbol_upper = symbol.upper()
            for key, beta in self.default_betas.items():
                if key in symbol_upper:
                    return beta
            
            # Default beta
            return 1.1
            
        except Exception as e:
            logger.error(f"Error estimating beta for {symbol}: {e}")
            return 1.0
    
    def _estimate_volatility(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Estimate asset volatility."""
        try:
            # Check if volatility is available in market data
            if market_data and symbol in market_data:
                symbol_data = market_data.get(symbol, {})
                if isinstance(symbol_data, dict) and 'volatility' in symbol_data:
                    vol = symbol_data['volatility']
                    if isinstance(vol, (int, float)) and 0 < vol < 10:
                        return vol
            
            # Use default volatilities
            symbol_upper = symbol.upper()
            for key, vol in self.default_volatilities.items():
                if key in symbol_upper:
                    return vol
            
            # Default volatility
            return 0.8
            
        except Exception as e:
            logger.error(f"Error estimating volatility for {symbol}: {e}")
            return 0.6
    
    def _estimate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        market_data: Dict[str, Any]
    ) -> float:
        """Estimate correlation between two assets."""
        if symbol1 == symbol2:
            return 1.0
        
        s1_upper = symbol1.upper()
        s2_upper = symbol2.upper()
        
        # Stablecoin correlations
        stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD']
        if any(stable in s1_upper for stable in stablecoins) and \
           any(stable in s2_upper for stable in stablecoins):
            return 0.95
        
        # BTC-ETH correlation
        if ('BTC' in s1_upper and 'ETH' in s2_upper) or \
           ('ETH' in s1_upper and 'BTC' in s2_upper):
            return 0.7
        
        # BTC with other crypto
        if 'BTC' in s1_upper or 'BTC' in s2_upper:
            return 0.6
        
        # Layer 1 correlations
        layer1s = ['ETH', 'SOL', 'AVAX', 'DOT', 'MATIC']
        if any(l1 in s1_upper for l1 in layer1s) and \
           any(l1 in s2_upper for l1 in layer1s):
            return 0.75
        
        # DeFi tokens
        defi = ['UNI', 'AAVE', 'LINK', 'SNX', 'YFI']
        if any(d in s1_upper for d in defi) and \
           any(d in s2_upper for d in defi):
            return 0.8
        
        # Default correlation
        return 0.65
    
    def _classify_sector(self, symbol: str) -> str:
        """Classify asset into sector."""
        symbol_upper = symbol.upper()
        
        if any(stable in symbol_upper for stable in ['USDT', 'USDC', 'DAI', 'BUSD']):
            return 'stablecoin'
        elif 'BTC' in symbol_upper:
            return 'bitcoin'
        elif any(l1 in symbol_upper for l1 in ['ETH', 'SOL', 'AVAX', 'DOT', 'MATIC']):
            return 'layer1'
        elif any(defi in symbol_upper for defi in ['UNI', 'AAVE', 'LINK', 'SNX', 'YFI']):
            return 'defi'
        elif any(meta in symbol_upper for meta in ['SAND', 'MANA', 'AXS', 'ENJ']):
            return 'metaverse'
        else:
            return 'other'
    
    def _get_correlation_group(self, symbol: str) -> str:
        """Get correlation group for the asset."""
        sector = self._classify_sector(symbol)
        
        # Map sectors to correlation groups
        correlation_groups = {
            'bitcoin': 'btc_group',
            'layer1': 'smart_contract',
            'defi': 'defi_group',
            'metaverse': 'gaming_meta',
            'stablecoin': 'stable_group',
            'other': 'alt_group'
        }
        
        return correlation_groups.get(sector, 'alt_group')
