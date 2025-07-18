"""
Health monitoring components for the trading bot.
Handles component health tracking, memory leak detection, and deadlock detection.
"""

import asyncio
import gc
import psutil
import threading
import weakref
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from collections import defaultdict

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ComponentHealthTracker:
    """Track component health and dependencies"""
    
    def __init__(self):
        self.component_status = {}
        self.component_dependencies = {}
        self.last_check_time = {}
        self.failure_counts = defaultdict(int)
        self.recovery_attempts = defaultdict(int)
        
    def register_component(self, name: str, dependencies: List[str] = None):
        """Register a component with its dependencies"""
        self.component_status[name] = 'initializing'
        self.component_dependencies[name] = dependencies or []
        self.last_check_time[name] = datetime.now()
        
    def update_status(self, name: str, status: str, error: str = None):
        """Update component status"""
        self.component_status[name] = status
        self.last_check_time[name] = datetime.now()
        
        if status == 'failed':
            self.failure_counts[name] += 1
            if error:
                logger.error(f"Component {name} failed: {error}")
        elif status == 'healthy':
            self.failure_counts[name] = 0
            self.recovery_attempts[name] = 0
            
    def check_dependencies(self, name: str) -> bool:
        """Check if all dependencies are healthy"""
        for dep in self.component_dependencies.get(name, []):
            if self.component_status.get(dep) != 'healthy':
                return False
        return True
        
    def get_cascade_impact(self, failed_component: str) -> Set[str]:
        """Get all components affected by a failure"""
        affected = {failed_component}
        
        # Find all components that depend on the failed one
        for comp, deps in self.component_dependencies.items():
            if failed_component in deps:
                affected.add(comp)
                # Recursively check dependencies
                affected.update(self.get_cascade_impact(comp))
                
        return affected


class MemoryLeakDetector:
    """Detect and handle memory leaks"""
    
    def __init__(self, threshold_mb: float = 100):
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.baseline_memory = None
        self.growth_history = []
        self.object_counts = {}
        
    def set_baseline(self):
        """Set memory baseline"""
        gc.collect()
        self.baseline_memory = psutil.Process().memory_info().rss
        self.object_counts = self._get_object_counts()
        
    def check_memory_growth(self) -> Dict[str, Any]:
        """Check for memory growth patterns"""
        gc.collect()
        current_memory = psutil.Process().memory_info().rss
        
        if self.baseline_memory is None:
            self.set_baseline()
            return {'status': 'baseline_set'}
            
        growth = current_memory - self.baseline_memory
        self.growth_history.append(growth)
        
        # Keep only recent history
        if len(self.growth_history) > 100:
            self.growth_history = self.growth_history[-100:]
            
        # Check for consistent growth
        if len(self.growth_history) >= 10:
            recent_growth = self.growth_history[-10:]
            avg_growth = sum(recent_growth) / len(recent_growth)
            
            if avg_growth > self.threshold_bytes:
                # Detect which objects are growing
                current_counts = self._get_object_counts()
                growing_objects = {}
                
                for obj_type, count in current_counts.items():
                    baseline_count = self.object_counts.get(obj_type, 0)
                    if count > baseline_count * 1.5:  # 50% growth
                        growing_objects[obj_type] = {
                            'baseline': baseline_count,
                            'current': count,
                            'growth': count - baseline_count
                        }
                        
                return {
                    'status': 'leak_detected',
                    'growth_bytes': growth,
                    'avg_growth_bytes': avg_growth,
                    'growing_objects': growing_objects
                }
                
        return {
            'status': 'normal',
            'growth_bytes': growth,
            'memory_mb': current_memory / (1024 * 1024)
        }
        
    def _get_object_counts(self) -> Dict[str, int]:
        """Get count of objects by type"""
        counts = defaultdict(int)
        for obj in gc.get_objects():
            counts[type(obj).__name__] += 1
        return dict(counts)
        
    def cleanup_memory(self, aggressive: bool = False):
        """Perform memory cleanup"""
        import torch
        import pandas as pd
        
        # Clear caches
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force garbage collection
        gc.collect()
        
        if aggressive:
            # Additional cleanup for known memory hogs
            # Clear matplotlib figures if any
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except ImportError:
                pass
                
            # Clear pandas option cache
            pd.reset_option('all')
            
            # Collect again
            gc.collect(2)  # Full collection


class DeadlockDetector:
    """Detect and recover from deadlocks"""
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        self.task_registry = weakref.WeakValueDictionary()
        self.task_start_times = {}
        self.lock = threading.Lock()
        
    def register_task(self, name: str, task: asyncio.Task):
        """Register a task for monitoring"""
        with self.lock:
            self.task_registry[name] = task
            self.task_start_times[name] = datetime.now()
            
    def unregister_task(self, name: str):
        """Unregister a completed task"""
        with self.lock:
            self.task_start_times.pop(name, None)
            
    def check_deadlocks(self) -> List[str]:
        """Check for potential deadlocks"""
        deadlocked = []
        current_time = datetime.now()
        
        with self.lock:
            for name, start_time in list(self.task_start_times.items()):
                if (current_time - start_time).total_seconds() > self.timeout:
                    task = self.task_registry.get(name)
                    if task and not task.done():
                        deadlocked.append(name)
                        
        return deadlocked
        
    async def recover_deadlock(self, task_name: str):
        """Attempt to recover from deadlock"""
        task = self.task_registry.get(task_name)
        if task and not task.done():
            logger.warning(f"Cancelling potentially deadlocked task: {task_name}")
            task.cancel()
            
            # Wait for cancellation
            try:
                await asyncio.wait_for(task, timeout=10)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
                
            # Unregister the task
            self.unregister_task(task_name)


class HealthMonitor:
    """Main health monitoring coordinator"""
    
    def __init__(self, config: dict, db_manager=None):
        self.config = config
        self.db = db_manager
        self.health_tracker = ComponentHealthTracker()
        self.memory_detector = MemoryLeakDetector(
            threshold_mb=config.get('monitoring.memory_growth_threshold_mb', 100)
        )
        self.deadlock_detector = DeadlockDetector(
            timeout=config.get('monitoring.task_timeout_seconds', 300)
        )
        self.network_partitions = defaultdict(int)
        self.partition_threshold = 5
        
    async def memory_monitoring_loop(self, running_flag, cleanup_callback=None):
        """Monitor memory usage and detect leaks"""
        check_interval = 60  # Check every minute
        cleanup_threshold_mb = self.config.get('monitoring.memory_cleanup_threshold_mb', 500)
        
        while running_flag:
            try:
                result = self.memory_detector.check_memory_growth()
                
                if result['status'] == 'leak_detected':
                    logger.warning(
                        f"Memory leak detected: {result['growth_bytes'] / (1024*1024):.2f} MB growth, "
                        f"Growing objects: {list(result['growing_objects'].keys())[:5]}"
                    )
                    
                    # Perform cleanup if memory usage is high
                    current_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    if current_memory_mb > cleanup_threshold_mb:
                        logger.info("Performing memory cleanup...")
                        self.memory_detector.cleanup_memory(aggressive=True)
                        
                        # Call cleanup callback if provided
                        if cleanup_callback:
                            await cleanup_callback()
                
                # Log memory status
                if result.get('memory_mb') and self.db:
                    self.db.record_system_metric('memory_usage_mb', result['memory_mb'])
                    
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(check_interval)
    
    async def deadlock_monitoring_loop(self, running_flag, restart_callback=None):
        """Monitor for deadlocked tasks"""
        check_interval = 30  # Check every 30 seconds
        
        while running_flag:
            try:
                deadlocked = self.deadlock_detector.check_deadlocks()
                
                if deadlocked:
                    logger.warning(f"Potentially deadlocked tasks detected: {deadlocked}")
                    
                    for task_name in deadlocked:
                        # Attempt recovery
                        await self.deadlock_detector.recover_deadlock(task_name)
                        
                        # Restart if it was a critical task
                        if task_name in ['trading_loop', 'risk_monitoring'] and restart_callback:
                            logger.info(f"Restarting critical task: {task_name}")
                            await restart_callback(task_name)
                            
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in deadlock monitoring: {e}")
                await asyncio.sleep(check_interval)
    
    async def component_health_loop(self, running_flag):
        """Monitor component health"""
        check_interval = 60  # Check every minute
        
        while running_flag:
            try:
                # Check each component
                for component in self.health_tracker.component_status:
                    if component in ['exchange', 'database', 'redis']:
                        # These have specific health checks
                        continue
                        
                    # Check if component has been failing repeatedly
                    if self.health_tracker.failure_counts[component] > 10:
                        logger.error(f"Component {component} has failed {self.health_tracker.failure_counts[component]} times")
                        
                        # Attempt recovery if not recently attempted
                        if self.health_tracker.recovery_attempts[component] < 3:
                            logger.info(f"Attempting to recover {component}")
                            self.health_tracker.recovery_attempts[component] += 1
                            # Component-specific recovery logic would go here
                            
                # Log overall health
                healthy = sum(
                    1 for status in self.health_tracker.component_status.values() 
                    if status == 'healthy'
                )
                total = len(self.health_tracker.component_status)
                
                if self.db:
                    self.db.record_system_metric('component_health_ratio', healthy / total if total > 0 else 0)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in component health monitoring: {e}")
                await asyncio.sleep(check_interval)
    
    async def check_network_partition(self, service: str, reconnect_callback=None) -> bool:
        """Check for network partition with a service"""
        if self.network_partitions[service] >= self.partition_threshold:
            logger.warning(f"Possible network partition detected with {service}")
            
            # Attempt to re-establish connection
            if reconnect_callback:
                try:
                    success = await reconnect_callback(service)
                    if success:
                        self.network_partitions[service] = 0
                        return True
                except Exception as e:
                    logger.error(f"Failed to reconnect to {service}: {e}")
                    return False
                    
        return True
    
    def increment_partition_count(self, service: str):
        """Increment network partition count for a service"""
        self.network_partitions[service] += 1
