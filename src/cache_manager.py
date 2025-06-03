"""
Cache Manager for RAG and Response Caching
Provides LRU caching with TTL support for optimizing LLM calls
"""
from typing import Any, Dict, Optional, Tuple, List
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
import asyncio
from functools import wraps

@dataclass
class CacheEntry:
    """Represents a cached item with metadata"""
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        """Update access count and timestamp for LRU"""
        self.access_count += 1

class CacheManager:
    """Thread-safe LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 500):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_memory_bytes = 0
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
        self._lock = asyncio.Lock()
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of cached value"""
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, (dict, list)):
            return len(json.dumps(value).encode('utf-8'))
        else:
            return 1024  # Default 1KB for unknown types
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired"""
        async with self._lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            entry = self.cache[key]
            
            if entry.is_expired():
                self.stats["expirations"] += 1
                del self.cache[key]
                self.total_memory_bytes -= entry.size_bytes
                return None
            
            # Move to end for LRU
            self.cache.move_to_end(key)
            entry.access()
            self.stats["hits"] += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: float):
        """Set value in cache with TTL"""
        async with self._lock:
            size = self._calculate_size(value)
            
            # Check if we need to evict items
            while (len(self.cache) >= self.max_size or 
                   self.total_memory_bytes + size > self.max_memory_bytes):
                if not self.cache:
                    break
                    
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                oldest_entry = self.cache[oldest_key]
                del self.cache[oldest_key]
                self.total_memory_bytes -= oldest_entry.size_bytes
                self.stats["evictions"] += 1
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size
            )
            self.cache[key] = entry
            self.total_memory_bytes += size
    
    async def clear_expired(self):
        """Remove all expired entries"""
        async with self._lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.cache[key]
                del self.cache[key]
                self.total_memory_bytes -= entry.size_bytes
                self.stats["expirations"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_entries": len(self.cache),
            "memory_usage_mb": self.total_memory_bytes / (1024 * 1024)
        }

# Global cache instances
rag_cache = CacheManager(max_size=500, max_memory_mb=200)
response_cache = CacheManager(max_size=300, max_memory_mb=100)
folder_cache = CacheManager(max_size=200, max_memory_mb=50)

def cache_rag_result(ttl: float = 300):
    """Decorator for caching RAG results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = rag_cache._generate_cache_key(*args, **kwargs)
            
            # Try to get from cache
            cached = await rag_cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await rag_cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def cache_response(ttl: float = 600):
    """Decorator for caching LLM responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = response_cache._generate_cache_key(*args, **kwargs)
            
            # Try to get from cache
            cached = await response_cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await response_cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Background task to clean expired entries
async def cleanup_caches_periodically():
    """Background task to clean up expired cache entries"""
    while True:
        await asyncio.sleep(60)  # Run every minute
        await rag_cache.clear_expired()
        await response_cache.clear_expired()
        await folder_cache.clear_expired() 