"""Tinygrad-style decorators replacing 403 verbose try/except blocks."""
import functools, asyncio, time, logging
from typing import TypeVar, Callable, Any, ParamSpec

P, T = ParamSpec('P'), TypeVar('T')
logger = logging.getLogger(__name__)

def safe_op(default: Any = None, log: bool = True, exc: type = Exception):
    """Replace verbose try/except with single decorator. Handles sync/async."""
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try: return await fn(*args, **kwargs)
            except exc as e:
                log and logger.error(f"{fn.__name__}: {e}")
                return default() if callable(default) else default
        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try: return fn(*args, **kwargs)
            except exc as e:
                log and logger.error(f"{fn.__name__}: {e}")
                return default() if callable(default) else default
        return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
    return decorator

def retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exc: type = Exception):
    """Retry with exponential backoff. Works with sync/async."""
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            d = delay
            for i in range(attempts):
                try: return await fn(*args, **kwargs)
                except exc as e:
                    if i == attempts - 1: raise
                    logger.warning(f"{fn.__name__} retry {i+1}/{attempts}: {e}")
                    await asyncio.sleep(d); d *= backoff
        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            d = delay
            for i in range(attempts):
                try: return fn(*args, **kwargs)
                except exc as e:
                    if i == attempts - 1: raise
                    logger.warning(f"{fn.__name__} retry {i+1}/{attempts}: {e}")
                    time.sleep(d); d *= backoff
        return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
    return decorator

def log_errors(fn: Callable[P, T]) -> Callable[P, T]:
    """Simple error logging without catching - just logs and re-raises."""
    @functools.wraps(fn)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try: return await fn(*args, **kwargs)
        except Exception as e: logger.exception(f"{fn.__name__} failed"); raise
    @functools.wraps(fn)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try: return fn(*args, **kwargs)
        except Exception as e: logger.exception(f"{fn.__name__} failed"); raise
    return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper

def timed(fn: Callable[P, T]) -> Callable[P, T]:
    """Log execution time for profiling."""
    @functools.wraps(fn)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        try: return await fn(*args, **kwargs)
        finally: logger.debug(f"{fn.__name__} took {time.perf_counter()-start:.3f}s")
    @functools.wraps(fn)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        try: return fn(*args, **kwargs)
        finally: logger.debug(f"{fn.__name__} took {time.perf_counter()-start:.3f}s")
    return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper

def cached(ttl: int = 300):
    """Simple TTL cache for expensive operations."""
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        cache: dict = {}
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache and now - cache[key][1] < ttl: return cache[key][0]
            result = fn(*args, **kwargs)
            cache[key] = (result, now)
            return result
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    return decorator

# Convenience: combine common patterns
def safe_retry(default: Any = None, attempts: int = 3, delay: float = 1.0):
    """Retry then fallback to default on final failure."""
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        return safe_op(default)(retry(attempts, delay)(fn))
    return decorator
