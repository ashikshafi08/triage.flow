"""Tinygrad-style async utilities replacing 25+ repetitive async patterns."""
import asyncio, functools, concurrent.futures
from typing import TypeVar, Callable, Coroutine, Any, ParamSpec

P, T = ParamSpec('P'), TypeVar('T')

def run_sync(coro: Coroutine[Any, Any, T], timeout: float = 30) -> T:
    """Run async code from sync context. Handles nested event loops."""
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(asyncio.run, coro).result(timeout)
    except RuntimeError:
        return asyncio.run(coro)

async def gather_safe(*coros: Coroutine, return_exceptions: bool = True) -> list:
    """asyncio.gather that doesn't fail on first exception."""
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)

def sync_to_async(fn: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """Decorator: make async function callable from sync code."""
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return run_sync(fn(*args, **kwargs))
    return wrapper

def async_to_sync(fn: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]:
    """Decorator: wrap sync function for async context (runs in thread pool)."""
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    return wrapper

async def async_retry(coro_fn: Callable[[], Coroutine], attempts: int = 3, delay: float = 1.0) -> Any:
    """Retry coroutine with backoff."""
    for i in range(attempts):
        try: return await coro_fn()
        except Exception as e:
            if i == attempts - 1: raise
            await asyncio.sleep(delay * (2 ** i))

async def timeout_or(coro: Coroutine, timeout: float, default: T = None) -> T:
    """Run coro with timeout, return default on timeout."""
    try: return await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError: return default

def fire_and_forget(coro: Coroutine) -> None:
    """Schedule coroutine without waiting for result."""
    try: asyncio.get_event_loop().create_task(coro)
    except RuntimeError: asyncio.run(coro)

# Parallel execution helpers
async def parallel_map(fn: Callable[[T], Coroutine], items: list[T], max_concurrent: int = 10) -> list:
    """Map async function over items with concurrency limit."""
    sem = asyncio.Semaphore(max_concurrent)
    async def limited(item):
        async with sem: return await fn(item)
    return await asyncio.gather(*[limited(i) for i in items])

async def first_success(*coros: Coroutine) -> Any:
    """Return first successful result, raise if all fail."""
    done, pending = await asyncio.wait([asyncio.create_task(c) for c in coros],
                                        return_when=asyncio.FIRST_COMPLETED)
    for p in pending: p.cancel()
    for d in done:
        if not d.exception(): return d.result()
    raise done.pop().exception()
