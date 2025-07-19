"""
View generator module for Black-Litterman optimization.
Generates investor views from ML models, technical analysis, and sentiment.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CryptoViewGenerator:
    """
    Generate investor views for Black-Litterman optimization.
    
    This class provides methods to generate views from multiple sources:
    - Machine learning model predictions
    - Technical analysis indicators
    - Sentiment analysis data
    
    Attributes:
        ml_models: Dict of ML models for each asset
        market_data: Market data for generating views
    """
    
    def __init__(self,
                 ml_models: Optional[Dict] = None,
                 market_data: Optional[Dict] = None):
        """
        Initialize the view generator.
        
        Args:
            ml_models: Dict of ML models {symbol: model}
            market_data: Dict of market data {symbol: DataFrame}
        """
        self.ml_models = ml_models or {}
        self.market_data = market_data
        
        # View generation parameters
        self.max_view_magnitude = 1.0  # Maximum 100% annual return view
        self.min_confidence = 0.1
        self.max_confidence = 0.95
    
    def generate_ml_views(self,
                         confidence_threshold: float = 0.6) -> Tuple[Dict, Dict]:
        """
        Generate views based on ML predictions.
        
        Args:
            confidence_threshold: Minimum confidence level to include a view
            
        Returns:
            Tuple of (views dict, confidences dict)
        """
        views = {}
        view_confidences = {}
        
        # Validate confidence threshold
        confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        
        for symbol, model in self.ml_models.items():
            try:
                # Get ML prediction
                if hasattr(model, 'predict') and self.market_data is not None:
                    symbol_data = self.market_data.get(symbol)
                    
                    if symbol_data is None or symbol_data.empty:
                        logger.warning(f"No data available for {symbol}")
                        continue
                    
                    prediction = model.predict(symbol_data)
                    
                    if isinstance(prediction, dict):
                        confidence = prediction.get('confidence', 0.5)
                        expected_return = prediction.get('expected_return', 0)
                        
                        # Validate prediction values
                        if not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
                            logger.warning(f"Invalid confidence for {symbol}: {confidence}")
                            continue
                            
                        if not isinstance(expected_return, (int, float)) or not np.isfinite(expected_return):
                            logger.warning(f"Invalid expected return for {symbol}: {expected_return}")
                            continue
                        
                        # Bound values
                        confidence = max(self.min_confidence, min(self.max_confidence, confidence))
                        expected_return = max(-self.max_view_magnitude, 
                                            min(self.max_view_magnitude, expected_return))
                        
                        if confidence >= confidence_threshold:
                            views[symbol] = expected_return
                            view_confidences[symbol] = confidence
                            
            except Exception as e:
                logger.warning(f"Error generating ML view for {symbol}: {e}")
                continue
        
        return views, view_confidences
    
    def generate_technical_views(self,
                                technical_data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Generate views based on technical analysis.
        
        Args:
            technical_data: DataFrame with technical indicators for each asset
            
        Returns:
            Tuple of (views dict, confidences dict)
        """
        views = {}
        view_confidences = {}
        
        # Validate input
        if technical_data is None or technical_data.empty:
            logger.warning("No technical data provided")
            return views, view_confidences
        
        for symbol in technical_data.columns:
            try:
                # Get technical indicators
                symbol_data = technical_data[symbol]
                
                if symbol_data.empty or symbol_data.isna().all():
                    continue
                
                # RSI-based view
                if 'rsi' in symbol_data:
                    rsi = symbol_data.get('rsi', 50)
                    
                    # Validate RSI
                    if not isinstance(rsi, (int, float)) or not 0 <= rsi <= 100:
                        logger.warning(f"Invalid RSI for {symbol}: {rsi}")
                        continue
                    
                    if rsi < 30:  # Oversold
                        views[symbol] = 0.05  # Expect 5% return
                        view_confidences[symbol] = min(self.max_confidence, (30 - rsi) / 30)
                    elif rsi > 70:  # Overbought
                        views[symbol] = -0.03  # Expect -3% return
                        view_confidences[symbol] = min(self.max_confidence, (rsi - 70) / 30)
                
                # MACD-based view enhancement
                if symbol in views and 'macd' in symbol_data:
                    macd = symbol_data.get('macd', 0)
                    
                    if isinstance(macd, (int, float)) and np.isfinite(macd):
                        if ((views[symbol] > 0 and macd > 0) or
                            (views[symbol] < 0 and macd < 0)):
                            view_confidences[symbol] = min(
                                self.max_confidence,
                                view_confidences[symbol] * 1.2
                            )
                
            except Exception as e:
                logger.warning(f"Error generating technical view for {symbol}: {e}")
                continue
        
        return views, view_confidences
    
    def generate_sentiment_views(self,
                               sentiment_data: Dict) -> Tuple[Dict, Dict]:
        """
        Generate views based on sentiment data.
        
        Args:
            sentiment_data: Dict of sentiment scores {symbol: score}
                           Scores should be between -1 and 1
            
        Returns:
            Tuple of (views dict, confidences dict)
        """
        views = {}
        view_confidences = {}
        
        # Validate input
        if not sentiment_data:
            logger.warning("No sentiment data provided")
            return views, view_confidences
        
        for symbol, sentiment_score in sentiment_data.items():
            try:
                # Validate sentiment score
                if not isinstance(sentiment_score, (int, float)) or not np.isfinite(sentiment_score):
                    logger.warning(f"Invalid sentiment score for {symbol}: {sentiment_score}")
                    continue
                
                # Bound sentiment score
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                
                # Convert sentiment to expected return
                if abs(sentiment_score) > 0.3:  # Only use strong sentiment
                    expected_return = sentiment_score * 0.08  # Max 8% expected return
                    confidence = min(self.max_confidence, abs(sentiment_score))
                    
                    views[symbol] = expected_return
                    view_confidences[symbol] = confidence
                    
            except Exception as e:
                logger.warning(f"Error generating sentiment view for {symbol}: {e}")
                continue
        
        return views, view_confidences
    
    def combine_views(self, *view_sets) -> Tuple[Dict, Dict]:
        """
        Combine multiple sets of views using confidence-weighted averaging.
        
        Args:
            *view_sets: Variable number of (views, confidences) tuples
            
        Returns:
            Combined (views dict, confidences dict)
        """
        combined_views = {}
        combined_confidences = {}
        
        # Validate inputs
        valid_view_sets = []
        for view_set in view_sets:
            if (isinstance(view_set, tuple) and len(view_set) == 2 and
                isinstance(view_set[0], dict) and isinstance(view_set[1], dict)):
                valid_view_sets.append(view_set)
            else:
                logger.warning("Invalid view set format, skipping")
        
        if not valid_view_sets:
            return combined_views, combined_confidences
        
        # Collect all views for each symbol
        symbol_views = {}
        symbol_confidences = {}
        
        for views, confidences in valid_view_sets:
            for symbol, view in views.items():
                try:
                    # Validate view and confidence
                    if symbol not in confidences:
                        logger.warning(f"No confidence for {symbol} view")
                        continue
                    
                    confidence = confidences[symbol]
                    
                    if (not isinstance(view, (int, float)) or not np.isfinite(view) or
                        not isinstance(confidence, (int, float)) or not np.isfinite(confidence)):
                        continue
                    
                    if symbol not in symbol_views:
                        symbol_views[symbol] = []
                        symbol_confidences[symbol] = []
                    
                    symbol_views[symbol].append(view)
                    symbol_confidences[symbol].append(confidence)
                    
                except Exception as e:
                    logger.warning(f"Error processing view for {symbol}: {e}")
                    continue
        
        # Combine views using confidence-weighted averaging
        for symbol in symbol_views:
            try:
                views_list = symbol_views[symbol]
                conf_list = symbol_confidences[symbol]
                
                if views_list and conf_list:
                    # Weighted average of views
                    total_conf = sum(conf_list)
                    if total_conf > 0:
                        weighted_view = sum(
                            v * c for v, c in zip(views_list, conf_list)
                        ) / total_conf
                        avg_confidence = total_conf / len(views_list)
                        
                        # Bound final values
                        weighted_view = max(-self.max_view_magnitude, 
                                          min(self.max_view_magnitude, weighted_view))
                        avg_confidence = max(self.min_confidence, 
                                           min(self.max_confidence, avg_confidence))
                        
                        combined_views[symbol] = weighted_view
                        combined_confidences[symbol] = avg_confidence
                        
            except Exception as e:
                logger.error(f"Error combining views for {symbol}: {e}")
                continue
        
        return combined_views, combined_confidences
    
    def generate_relative_views(self,
                               asset1: str,
                               asset2: str,
                               expected_outperformance: float,
                               confidence: float) -> Tuple[np.ndarray, float, float]:
        """
        Generate a relative view (asset1 will outperform asset2).
        
        Args:
            asset1: Symbol of the outperforming asset
            asset2: Symbol of the underperforming asset
            expected_outperformance: Expected outperformance (e.g., 0.05 for 5%)
            confidence: Confidence in the view (0-1)
            
        Returns:
            Tuple of (picking_vector, view_return, view_confidence)
        """
        # This is a placeholder for relative view generation
        # In practice, this would create a picking vector with +1 for asset1 and -1 for asset2
        logger.info(f"Generating relative view: {asset1} to outperform {asset2} by {expected_outperformance}")
        
        # Validate inputs
        confidence = max(self.min_confidence, min(self.max_confidence, confidence))
        expected_outperformance = max(-self.max_view_magnitude, 
                                    min(self.max_view_magnitude, expected_outperformance))
        
        return None, expected_outperformance, confidence
