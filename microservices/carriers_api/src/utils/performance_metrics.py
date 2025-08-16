import time
import psutil
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    items_processed: int = 0
    errors: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def finalize(self, end_time: Optional[float] = None):
        """Finalize metrics calculation"""
        if end_time is None:
            end_time = time.time()
        
        self.end_time = end_time
        self.duration = end_time - self.start_time
        
        if self.memory_before is not None:
            self.memory_after = self._get_memory_usage()
    
    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for logging"""
        return {
            'operation': self.operation,
            'duration_s': self.duration,
            'memory_before_mb': self.memory_before,
            'memory_after_mb': self.memory_after,
            'memory_peak_mb': self.memory_peak,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'items_processed': self.items_processed,
            'throughput_items_per_second': self.items_processed / max(0.001, self.duration or 0.001),
            'errors': self.errors,
            **self.metadata
        }


class PerformanceTracker:
    """Thread-safe performance tracking system"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self._metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self._active_metrics: Dict[str, PerformanceMetrics] = {}
    
    @contextmanager
    def track_operation(self, operation: str, **metadata):
        """Context manager to track an operation's performance"""
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            memory_before=PerformanceMetrics._get_memory_usage(),
            metadata=metadata
        )
        
        with self._lock:
            self._active_metrics[operation] = metrics
        
        try:
            yield metrics
        finally:
            metrics.finalize()
            
            with self._lock:
                self._metrics.append(metrics)
                self._active_metrics.pop(operation, None)
            
            # Log metrics
            if logger.isEnabledFor(self.log_level):
                self._log_metrics(metrics)
    
    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        msg_parts = [f"{metrics.operation}: {metrics.duration:.2f}s"]
        
        if metrics.items_processed > 0:
            throughput = metrics.items_processed / metrics.duration
            msg_parts.append(f"({metrics.items_processed} items, {throughput:.1f} items/s)")
        
        if metrics.memory_before and metrics.memory_after:
            memory_change = metrics.memory_after - metrics.memory_before
            if abs(memory_change) > 1:  # Only show if > 1MB change
                msg_parts.append(f"Memory: {memory_change:+.1f}MB")
        
        if metrics.cache_hits + metrics.cache_misses > 0:
            hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
            msg_parts.append(f"Cache: {hit_rate:.1%} hit rate")
        
        if metrics.errors > 0:
            msg_parts.append(f"Errors: {metrics.errors}")
        
        logger.log(self.log_level, " | ".join(msg_parts))
    
    def record_cache_hit(self, operation: str):
        """Record a cache hit for the given operation"""
        with self._lock:
            if operation in self._active_metrics:
                self._active_metrics[operation].cache_hits += 1
    
    def record_cache_miss(self, operation: str):
        """Record a cache miss for the given operation"""
        with self._lock:
            if operation in self._active_metrics:
                self._active_metrics[operation].cache_misses += 1
    
    def record_items_processed(self, operation: str, count: int):
        """Record number of items processed for the given operation"""
        with self._lock:
            if operation in self._active_metrics:
                self._active_metrics[operation].items_processed += count
    
    def record_error(self, operation: str):
        """Record an error for the given operation"""
        with self._lock:
            if operation in self._active_metrics:
                self._active_metrics[operation].errors += 1
    
    def get_metrics(self, operation: str = None) -> List[Dict]:
        """Get performance metrics"""
        with self._lock:
            metrics = self._metrics.copy()
        
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        return [m.to_dict() for m in metrics]
    
    def get_summary(self) -> Dict:
        """Get performance summary statistics"""
        with self._lock:
            metrics = self._metrics.copy()
        
        if not metrics:
            return {}
        
        # Group by operation
        by_operation = {}
        for metric in metrics:
            op = metric.operation
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(metric)
        
        summary = {}
        for operation, op_metrics in by_operation.items():
            durations = [m.duration for m in op_metrics if m.duration]
            throughputs = [
                m.items_processed / m.duration 
                for m in op_metrics 
                if m.duration and m.items_processed > 0
            ]
            
            summary[operation] = {
                'count': len(op_metrics),
                'total_duration_s': sum(durations),
                'avg_duration_s': sum(durations) / len(durations) if durations else 0,
                'min_duration_s': min(durations) if durations else 0,
                'max_duration_s': max(durations) if durations else 0,
                'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
                'total_items': sum(m.items_processed for m in op_metrics),
                'total_cache_hits': sum(m.cache_hits for m in op_metrics),
                'total_cache_misses': sum(m.cache_misses for m in op_metrics),
                'total_errors': sum(m.errors for m in op_metrics)
            }
        
        return summary
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        with self._lock:
            self._metrics.clear()
    
    def log_summary(self):
        """Log performance summary"""
        summary = self.get_summary()
        
        if not summary:
            logger.info("No performance metrics recorded")
            return
        
        logger.info("=== Performance Summary ===")
        
        for operation, stats in summary.items():
            lines = [f"{operation}:"]
            lines.append(f"  Executions: {stats['count']}")
            lines.append(f"  Total time: {stats['total_duration_s']:.2f}s")
            lines.append(f"  Avg time: {stats['avg_duration_s']:.2f}s")
            
            if stats['total_items'] > 0:
                lines.append(f"  Total items: {stats['total_items']}")
                lines.append(f"  Avg throughput: {stats['avg_throughput']:.1f} items/s")
            
            if stats['total_cache_hits'] + stats['total_cache_misses'] > 0:
                total_requests = stats['total_cache_hits'] + stats['total_cache_misses']
                hit_rate = stats['total_cache_hits'] / total_requests
                lines.append(f"  Cache hit rate: {hit_rate:.1%}")
            
            if stats['total_errors'] > 0:
                lines.append(f"  Errors: {stats['total_errors']}")
            
            logger.info("\n".join(lines))


# Global performance tracker instance
_global_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    return _global_tracker


def track_performance(operation: str, **metadata):
    """Decorator/context manager for tracking performance"""
    return _global_tracker.track_operation(operation, **metadata)


def record_cache_hit(operation: str):
    """Record cache hit for global tracker"""
    _global_tracker.record_cache_hit(operation)


def record_cache_miss(operation: str):
    """Record cache miss for global tracker"""  
    _global_tracker.record_cache_miss(operation)


def record_items_processed(operation: str, count: int):
    """Record items processed for global tracker"""
    _global_tracker.record_items_processed(operation, count)


def log_performance_summary():
    """Log performance summary for global tracker"""
    _global_tracker.log_summary()