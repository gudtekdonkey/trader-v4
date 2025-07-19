"""
Volume profile module for VWAP calculations.
Handles historical volume data processing and scheduling.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class VolumeProfileManager:
    """Manages volume profile data for VWAP execution"""
    
    def __init__(self, executor):
        self.executor = executor
        
    async def get_volume_profile_safe(self, symbol: str, days: int = 20) -> Optional[List[Dict]]:
        """
        Get historical volume profile for VWAP calculation with error handling
        
        Args:
            symbol: Trading symbol
            days: Number of days of historical data to use
            
        Returns:
            Volume profile data or None if unavailable
        """
        try:
            # Try to get real volume data from exchange
            volume_profile = None
            
            try:
                # This would typically fetch real volume data
                historical_data = await self.executor.exchange.get_historical_volume(
                    symbol, days
                )
                
                if historical_data:
                    volume_profile = self.process_volume_data(historical_data)
                    
            except Exception as e:
                logger.warning(f"Failed to get real volume data: {e}")
            
            # Use fallback if no real data
            if not volume_profile:
                logger.info("Using default volume profile")
                volume_profile = self._get_default_volume_profile()
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Error getting volume profile: {e}")
            return None
    
    def process_volume_data(self, historical_data: List[Dict]) -> List[Dict]:
        """
        Process raw volume data into hourly profile
        
        Args:
            historical_data: Raw historical volume data
            
        Returns:
            Processed hourly volume profile
        """
        try:
            # Group by hour and calculate average volumes
            hourly_volumes = {}
            
            for data_point in historical_data:
                if 'timestamp' in data_point and 'volume' in data_point:
                    hour = data_point['timestamp'].hour
                    volume = data_point.get('volume', 0)
                    
                    if hour not in hourly_volumes:
                        hourly_volumes[hour] = []
                    hourly_volumes[hour].append(volume)
            
            # Calculate average and weights
            total_volume = 0
            profile = []
            
            for hour in range(24):
                if hour in hourly_volumes:
                    avg_volume = np.mean(hourly_volumes[hour])
                else:
                    avg_volume = 0
                    
                total_volume += avg_volume
                profile.append({
                    'hour': hour,
                    'avg_volume': avg_volume
                })
            
            # Calculate weights
            if total_volume > 0:
                for item in profile:
                    item['volume_weight'] = item['avg_volume'] / total_volume
            else:
                # Equal weights if no volume data
                for item in profile:
                    item['volume_weight'] = 1.0 / 24
                    
            return profile
            
        except Exception as e:
            logger.error(f"Error processing volume data: {e}")
            return []
    
    def _get_default_volume_profile(self) -> List[Dict]:
        """Get default volume profile when real data unavailable"""
        # Typical crypto volume profile (higher during certain hours)
        hours = list(range(24))
        
        # Mock volume weights (higher volume during active trading hours)
        volume_weights = [
            0.02, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5
            0.04, 0.06, 0.08, 0.09, 0.08, 0.07,  # 6-11
            0.06, 0.08, 0.09, 0.08, 0.07, 0.06,  # 12-17
            0.05, 0.04, 0.03, 0.03, 0.02, 0.02   # 18-23
        ]
        
        profile = []
        for hour, weight in zip(hours, volume_weights):
            profile.append({
                'hour': hour,
                'volume_weight': weight,
                'avg_volume': weight * 1000000  # Mock volume
            })
        
        return profile
    
    def calculate_vwap_schedule(self, 
                               total_size: float, 
                               duration_minutes: int, 
                               volume_profile: List[Dict]) -> List[Dict]:
        """
        Calculate VWAP execution schedule based on volume profile
        
        Args:
            total_size: Total size to execute
            duration_minutes: Duration for execution
            volume_profile: Historical volume profile
            
        Returns:
            List of scheduled execution slices
        """
        try:
            current_hour = datetime.now().hour
            
            # Find relevant volume weights for execution period
            execution_hours = max(1, duration_minutes // 60 + 1)
            relevant_weights = []
            
            for i in range(execution_hours):
                hour = (current_hour + i) % 24
                hour_profile = next((p for p in volume_profile if p['hour'] == hour), None)
                if hour_profile:
                    relevant_weights.append(hour_profile['volume_weight'])
                else:
                    relevant_weights.append(1.0 / 24)  # Equal weight fallback
            
            # Validate weights
            if not relevant_weights or all(w == 0 for w in relevant_weights):
                logger.warning("Invalid volume weights, using equal distribution")
                relevant_weights = [1.0 / execution_hours] * execution_hours
            
            # Normalize weights
            total_weight = sum(relevant_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in relevant_weights]
            else:
                normalized_weights = [1.0 / len(relevant_weights)] * len(relevant_weights)
            
            # Create schedule
            schedule = []
            interval_minutes = max(1, duration_minutes // len(normalized_weights))
            
            for i, weight in enumerate(normalized_weights):
                slice_size = total_size * weight
                
                # Ensure minimum size
                if slice_size >= self.executor.min_order_size:
                    schedule.append({
                        'slice_number': i + 1,
                        'size': slice_size,
                        'wait_minutes': interval_minutes if i > 0 else 0,
                        'volume_weight': weight
                    })
            
            # If no valid slices, create at least one
            if not schedule:
                schedule = [{
                    'slice_number': 1,
                    'size': total_size,
                    'wait_minutes': 0,
                    'volume_weight': 1.0
                }]
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error calculating VWAP schedule: {e}")
            # Fallback to equal distribution
            n_slices = max(1, duration_minutes // 10)
            slice_size = total_size / n_slices
            return [{'slice_number': i+1, 'size': slice_size, 'wait_minutes': 10 if i > 0 else 0} 
                   for i in range(n_slices)]
    
    def calculate_iceberg_wait_time(self, symbol: str, side: str, failures: int) -> float:
        """
        Calculate dynamic wait time for iceberg orders based on market conditions
        
        Args:
            symbol: Trading symbol
            side: Order side
            failures: Number of consecutive failures
            
        Returns:
            Wait time in seconds
        """
        try:
            # Base wait time
            base_wait = 5  # seconds
            
            # Increase wait time with failures (exponential backoff)
            failure_factor = min(2 ** failures, 8)
            
            # Could add market-based adjustments here
            # For example, increase wait time during high volatility
            
            wait_time = base_wait * failure_factor
            
            # Cap maximum wait time
            return min(wait_time, 60)  # Max 1 minute
            
        except Exception as e:
            logger.error(f"Error calculating wait time: {e}")
            return 10  # Default 10 seconds
