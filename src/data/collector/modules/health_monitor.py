"""
Connection health monitoring
"""

from datetime import datetime
from typing import Dict, Any
from collections import deque


class ConnectionHealth:
    """Track connection health metrics"""
    
    def __init__(self, window_size: int = 100):
        self.message_times = deque(maxlen=window_size)
        self.error_times = deque(maxlen=window_size)
        self.last_message_time = None
        self.connection_start = datetime.now()
        self.reconnect_count = 0
    
    def record_message(self):
        """Record successful message"""
        now = datetime.now()
        self.message_times.append(now)
        self.last_message_time = now
    
    def record_error(self):
        """Record error"""
        self.error_times.append(datetime.now())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get health metrics"""
        now = datetime.now()
        
        # Message rate
        if len(self.message_times) >= 2:
            time_span = (self.message_times[-1] - self.message_times[0]).total_seconds()
            message_rate = len(self.message_times) / time_span if time_span > 0 else 0
        else:
            message_rate = 0
        
        # Error rate
        recent_errors = sum(1 for t in self.error_times if (now - t).total_seconds() < 60)
        
        # Latency
        latency = (now - self.last_message_time).total_seconds() if self.last_message_time else float('inf')
        
        return {
            'message_rate': message_rate,
            'recent_errors': recent_errors,
            'latency': latency,
            'reconnect_count': self.reconnect_count,
            'uptime': (now - self.connection_start).total_seconds()
        }
