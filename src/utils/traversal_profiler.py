# traversal_profiler.py
"""
Lightweight micro-profiler for traversal operations.
Provides context manager and decorator-based profiling with minimal code changes.
"""
import time
from contextlib import contextmanager
from functools import wraps


class TraversalProfiler:
    """
    Lightweight profiler for measuring execution time of code sections.

    Usage:
        profiler = TraversalProfiler()
        profiler.start('encoding')
        # ... code ...
        profiler.stop('encoding')
        profiler.log_summary()
    """

    def __init__(self, enabled: bool = True, level: str = 'full'):
        self.enabled = enabled
        self.level = level  # 'off', 'coarse', 'full'
        self._timings = {}  # key -> aggregate statistics
        self._starts = {}   # key -> stack of start times
        self._active_frames = []  # стек активных таймеров: key, start, child_ms
        self._previous_totals = {}
        self._traversal_nodes = 0
        self.coarse_keys = {
            'traverse_total', 'clone_state', 'encoding', 'recursive_children',
            'regret_compute', 'sampling_policy', 'legal_actions', 'network_forward',
            'reservoir_write', 'buffer_append', 'encoding_opponent',
            'network_forward_opponent',
            'network_forward_eval', 'encoding_eval',
            'legal_actions_eval',
        }

    def set_level(self, level: str):
        self.level = level
        if level == 'off':
            self.enabled = False
        else:
            self.enabled = True

    def start(self, key: str) -> None:
        """Start timing a section. Рекурсивные вложенные вызовы поддерживаются стеком."""
        if not self.enabled:
            return
        if self.level == 'coarse' and key not in self.coarse_keys:
            return
        started_at = time.perf_counter()
        self._starts.setdefault(key, []).append(started_at)
        self._active_frames.append([key, started_at, 0.0])

    def stop(self, key: str) -> float:
        """Stop timing a section and record the duration. Returns duration in ms."""
        if not self.enabled:
            return 0.0
        if self.level == 'coarse' and key not in self.coarse_keys:
            return 0.0
        stack = self._starts.get(key)
        if not stack or not self._active_frames:
            return 0.0

        frame_key, started_at, child_ms = self._active_frames[-1]
        if frame_key != key:
            raise RuntimeError(
                f"Profiler timer order violation: expected {frame_key}, got {key}"
            )
        self._active_frames.pop()
        duration = (time.perf_counter() - started_at) * 1000  # ms
        exclusive_ms = max(duration - child_ms, 0.0)
        if self._active_frames:
            self._active_frames[-1][2] += duration
        stack.pop()
        if not stack:
            del self._starts[key]

        stats = self._timings.setdefault(key, {
            'count': 0,
            'total_ms': 0.0,
            'exclusive_ms': 0.0,
            'min_ms': float('inf'),
            'max_ms': 0.0,
        })
        stats['count'] += 1
        stats['total_ms'] += duration
        stats['exclusive_ms'] += exclusive_ms
        stats['min_ms'] = min(stats['min_ms'], duration)
        stats['max_ms'] = max(stats['max_ms'], duration)
        return duration

    def reset(self) -> None:
        """Reset current iteration timings and retain previous totals for growth."""
        self._previous_totals = {
            key: stats['total_ms'] for key, stats in self._timings.items()
        }
        self._timings.clear()
        self._starts.clear()
        self._active_frames.clear()
        self._traversal_nodes = 0

    def assert_balanced(self) -> None:
        if self._starts or self._active_frames:
            raise RuntimeError(f"Profiler timer leak: {self._starts}")

    def set_traversal_nodes(self, count: int) -> None:
        self._traversal_nodes = max(int(count), 0)

    def get_stats(self, key: str) -> dict:
        """Get statistics for a specific key."""
        stats = self._timings.get(key)
        if not stats:
            return {
                'count': 0, 'total_ms': 0.0, 'exclusive_ms': 0.0,
                'avg_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0,
            }
        count = stats['count']
        return {
            'count': count,
            'total_ms': stats['total_ms'],
            'exclusive_ms': stats['exclusive_ms'],
            'avg_ms': stats['total_ms'] / count if count else 0.0,
            'min_ms': stats['min_ms'] if count else 0.0,
            'max_ms': stats['max_ms'] if count else 0.0,
        }

    def get_all_stats(self) -> dict:
        """Get statistics for all keys."""
        return {key: self.get_stats(key) for key in self._timings}

    def _nesting_map(self) -> dict:
        return {
            'traverser_postprocess': {
                'action_ev_compute', 'action_regret_normalize', 'advantage_buffer_write',
                'strategy_buffer_write',
            },
            'advantage_buffer_write': {'buffer_append'},
        }

    def _leaf_keys(self):
        parents = set(self._nesting_map())
        excluded = {
            'traverse_total', 'network_forward_eval', 'encoding_eval', 'legal_actions_eval',
            'recursive_children',
        }
        return [key for key in self._timings if key not in excluded and key not in parents]

    def leaf_total_ms(self) -> float:
        return sum(self._timings[key]['total_ms'] for key in self._leaf_keys())

    def total_ms(self) -> float:
        """Суммарное время листовых секций, пригодное для TimeBudget."""
        return self.leaf_total_ms()

    def traversal_self_ms(self) -> float:
        """Непокрытое таймерами время корневых traversals."""
        return self.get_stats('traverse_total')['exclusive_ms']

    def profiled_traversal_ms(self) -> float:
        total = self.get_stats('traverse_total')['total_ms']
        return max(total - self.traversal_self_ms(), 0.0)

    def log_summary(self, logger=None) -> str:
        """
        Log a summary of all timings.
        Returns the summary string.
        """
        if not self._timings:
            msg = "[TraversalProfiler] No timings recorded."
            if logger:
                logger(msg)
            else:
                print(msg)
            return msg

        lines = ["[TraversalProfiler] Summary:"]
        lines.append("-" * 110)
        lines.append(f"{'Section':<25} {'Count':>8} {'Total(ms)':>12} {'Self(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'us/node':>10} {'growth':>10}")
        lines.append("-" * 110)

        sorted_keys = sorted(self._timings.keys(),
                             key=lambda k: self._timings[k]['total_ms'],
                             reverse=True)
        for key in sorted_keys:
            stats = self.get_stats(key)
            previous = self._previous_totals.get(key)
            growth = (stats['total_ms'] / previous) if previous and previous > 0 else 0.0
            us_per_node = (stats['total_ms'] * 1000.0 / self._traversal_nodes
                           if self._traversal_nodes else 0.0)
            lines.append(
                f"{key:<25} {stats['count']:>8} {stats['total_ms']:>12.2f} {stats['exclusive_ms']:>12.2f} "
                f"{stats['avg_ms']:>10.3f} {stats['min_ms']:>10.3f} {stats['max_ms']:>10.3f} "
                f"{us_per_node:>10.3f} {growth:>9.2f}x"
            )

        lines.append("-" * 110)
        lines.append(f"{'TOTAL (covered traversal)':<25} {'':<8} {self.profiled_traversal_ms():>12.2f}")

        traverse_total_ms = self.get_stats('traverse_total')['total_ms']
        if traverse_total_ms > 0:
            self_ms = self.traversal_self_ms()
            covered_ms = self.profiled_traversal_ms()
            pct = self_ms / traverse_total_ms * 100.0
            lines.append(
                f"{'BALANCE':<25} traverse_total={traverse_total_ms:.2f}ms "
                f"covered={covered_ms:.2f}ms traverse_self={self_ms:.2f}ms ({pct:.1f}%)"
            )
            lines.append(
                f"{'traverse_self':<25} {'':<8} {self_ms:>12.2f}  ({pct:.1f}% of traverse_total)"
            )

        summary = "\n".join(lines)
        if logger:
            logger(summary)
        else:
            print(summary)
        return summary


@contextmanager
def profile_section(profiler: TraversalProfiler, key: str):
    """
    Context manager for profiling a code section.

    Usage:
        with profile_section(profiler, 'encoding'):
            # ... code to profile ...
    """
    profiler.start(key)
    try:
        yield
    finally:
        profiler.stop(key)


def profile_method(key: str = None):
    """
    Decorator for profiling a method.
    Uses TRAVERSAL_PROFILER global instance.

    Usage:
        @profile_method('my_function')
        def my_function():
            ...
    """
    def decorator(func):
        nonlocal key
        if key is None:
            key = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            TRAVERSAL_PROFILER.start(key)
            try:
                return func(*args, **kwargs)
            finally:
                TRAVERSAL_PROFILER.stop(key)
        return wrapper
    return decorator


# Global instance for easy access
TRAVERSAL_PROFILER = TraversalProfiler(enabled=False, level='full')


def enable_traversal_profiler():
    """Enable the global traversal profiler."""
    TRAVERSAL_PROFILER.enabled = True


def disable_traversal_profiler():
    """Disable the global traversal profiler."""
    TRAVERSAL_PROFILER.enabled = False


def set_traversal_profiler_level(level: str):
    """Set the profiling level: 'off', 'coarse', or 'full'."""
    TRAVERSAL_PROFILER.set_level(level)
