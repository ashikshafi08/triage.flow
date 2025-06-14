"""
Enhanced Cache Manager with Redis Support
Optimized for AI/RAG workloads with multi-layer caching strategy
"""
import redis.asyncio as redis
from typing import Optional, Any, Dict, List, Union
import json
import hashlib
import time
import logging
import asyncio
from functools import wraps
from enum import Enum
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)

class CacheLayer(Enum):
    """Cache layer types ordered by speed"""
    L1_MEMORY = "memory"      # Fastest, smallest capacity
    L2_REDIS = "redis"        # Fast, shared across instances
    L3_PERSISTENT = "disk"    # Slowest, largest capacity

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    size_bytes: int = 0
    layer: CacheLayer = CacheLayer.L1_MEMORY
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        self.access_count += 1

class RedisManager:
    """Singleton Redis connection manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.redis_client = None
            self.initialized = False
            self.connection_pool = None
    
    async def initialize(self):
        """Initialize Redis connection with your project's config"""
        if self.initialized:
            return
            
        try:
            redis_config = settings.REDIS_CONFIG
            redis_url = f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
            
            self.connection_pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=50,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.initialized = True
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis unavailable, falling back to memory-only cache: {e}")
            self.initialized = False
    
    async def close(self):
        """Cleanup Redis connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()

class EnhancedCacheManager:
    """
    Multi-layer cache manager optimized for AI/RAG workloads
    Maintains compatibility with existing cache_manager.py interface
    """
    
    def __init__(self, 
                 namespace: str,
                 max_memory_size: int = 100 * 1024 * 1024,  # 100MB L1 cache
                 default_ttl: int = 3600,
                 redis_manager: Optional[RedisManager] = None):
        self.namespace = namespace
        self.max_memory_size = max_memory_size
        self.default_ttl = default_ttl
        self.redis = redis_manager or RedisManager()
        
        # L1 Memory cache (compatible with existing interface)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_size = 0
        self.stats = {
            "hits": 0, "misses": 0, "evictions": 0, "expirations": 0,
            "l1_hits": 0, "l2_hits": 0, "l1_sets": 0, "l2_sets": 0
        }
        self._lock = asyncio.Lock()
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key (compatible with existing interface)"""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        hash_key = hashlib.sha256(key_str.encode()).hexdigest()
        return f"{self.namespace}:{hash_key}"
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return 1024  # 1KB default
        except:
            return 1024
    
    async def get(self, key: str, layers: List[CacheLayer] = None) -> Optional[Any]:
        """Get value from cache layers in order"""
        if not settings.CACHE_ENABLED:
            return None
            
        if layers is None:
            layers = [CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS]
        
        full_key = f"{self.namespace}:{key}" if not key.startswith(self.namespace) else key
        
        # L1 Memory Cache
        if CacheLayer.L1_MEMORY in layers:
            async with self._lock:
                if full_key in self.memory_cache:
                    entry = self.memory_cache[full_key]
                    if not entry.is_expired():
                        entry.access()
                        self.stats["hits"] += 1
                        self.stats["l1_hits"] += 1
                        return entry.value
                    else:
                        # Remove expired entry
                        self.memory_size -= entry.size_bytes
                        del self.memory_cache[full_key]
                        self.stats["expirations"] += 1
        
        # L2 Redis Cache
        if CacheLayer.L2_REDIS in layers and self.redis.initialized:
            try:
                cached_data = await self.redis.redis_client.get(full_key)
                if cached_data:
                    data = json.loads(cached_data)
                    self.stats["hits"] += 1
                    self.stats["l2_hits"] += 1
                    
                    # Promote to L1 if there's space
                    if CacheLayer.L1_MEMORY in layers:
                        await self._set_memory_cache(full_key, data["value"], data["ttl"])
                    
                    return data["value"]
            except Exception as e:
                logger.warning(f"Redis get error for key {full_key}: {e}")
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, 
                  key: str, 
                  value: Any, 
                  ttl: Optional[int] = None,
                  layers: List[CacheLayer] = None):
        """Set value in specified cache layers"""
        if not settings.CACHE_ENABLED:
            return
            
        if ttl is None:
            ttl = self.default_ttl
        if layers is None:
            layers = [CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS]
        
        full_key = f"{self.namespace}:{key}" if not key.startswith(self.namespace) else key
        
        # Set in L1 Memory
        if CacheLayer.L1_MEMORY in layers:
            await self._set_memory_cache(full_key, value, ttl)
            self.stats["l1_sets"] += 1
        
        # Set in L2 Redis
        if CacheLayer.L2_REDIS in layers and self.redis.initialized:
            try:
                cache_data = {
                    "value": value,
                    "ttl": ttl,
                    "timestamp": time.time(),
                    "namespace": self.namespace
                }
                await self.redis.redis_client.setex(
                    full_key, 
                    ttl, 
                    json.dumps(cache_data, default=str)
                )
                self.stats["l2_sets"] += 1
            except Exception as e:
                logger.warning(f"Redis set error for key {full_key}: {e}")
    
    async def _set_memory_cache(self, key: str, value: Any, ttl: int):
        """Set value in L1 memory cache with LRU eviction"""
        async with self._lock:
            size = self._estimate_size(value)
            
            # Evict if necessary (LRU)
            while (len(self.memory_cache) > 0 and 
                   (self.memory_size + size > self.max_memory_size or 
                    len(self.memory_cache) >= settings.MAX_CACHE_SIZE)):
                
                # Find least recently used
                lru_key = min(self.memory_cache.keys(), 
                             key=lambda k: self.memory_cache[k].access_count)
                lru_entry = self.memory_cache[lru_key]
                self.memory_size -= lru_entry.size_bytes
                del self.memory_cache[lru_key]
                self.stats["evictions"] += 1
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size,
                layer=CacheLayer.L1_MEMORY
            )
            self.memory_cache[key] = entry
            self.memory_size += size
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # Memory cache
        async with self._lock:
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_delete:
                entry = self.memory_cache[key]
                self.memory_size -= entry.size_bytes
                del self.memory_cache[key]
        
        # Redis cache
        if self.redis.initialized:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self.redis.redis_client.scan(
                        cursor, match=f"*{pattern}*", count=100
                    )
                    if keys:
                        await self.redis.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis pattern invalidation error: {e}")
    
    async def clear_expired(self):
        """Remove expired entries from memory cache"""
        async with self._lock:
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.memory_cache[key]
                self.memory_size -= entry.size_bytes
                del self.memory_cache[key]
                self.stats["expirations"] += 1
    
    async def delete(self, key: str) -> bool:
        """Delete a specific cache entry from all layers"""
        full_key = f"{self.namespace}:{key}" if not key.startswith(self.namespace) else key
        deleted = False
        
        # Delete from memory cache
        async with self._lock:
            if full_key in self.memory_cache:
                entry = self.memory_cache[full_key]
                self.memory_size -= entry.size_bytes
                del self.memory_cache[full_key]
                deleted = True
        
        # Delete from Redis cache
        if self.redis.initialized:
            try:
                result = await self.redis.redis_client.delete(full_key)
                if result > 0:
                    deleted = True
            except Exception as e:
                logger.warning(f"Redis delete error for key {full_key}: {e}")
        
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "l1_hit_rate": self.stats["l1_hits"] / total_requests if total_requests > 0 else 0,
            "l2_hit_rate": self.stats["l2_hits"] / total_requests if total_requests > 0 else 0,
            "total_entries": len(self.memory_cache),
            "memory_usage_mb": self.memory_size / (1024 * 1024),
            "redis_available": self.redis.initialized,
            "namespace": self.namespace
        }

# Global instances (maintaining compatibility)
redis_manager = RedisManager()

# Enhanced cache instances optimized for AI workloads
rag_cache = EnhancedCacheManager(
    namespace="rag",
    max_memory_size=200 * 1024 * 1024,  # 200MB for RAG contexts
    default_ttl=settings.CACHE_TTL_RAG,
    redis_manager=redis_manager
)

response_cache = EnhancedCacheManager(
    namespace="llm_response", 
    max_memory_size=100 * 1024 * 1024,  # 100MB for LLM responses
    default_ttl=settings.CACHE_TTL_RESPONSE,
    redis_manager=redis_manager
)

folder_cache = EnhancedCacheManager(
    namespace="folder",
    max_memory_size=50 * 1024 * 1024,   # 50MB for folder structures
    default_ttl=settings.CACHE_TTL_FOLDER,
    redis_manager=redis_manager
)

# AI-specific caches
issue_cache = EnhancedCacheManager(
    namespace="issues",
    max_memory_size=150 * 1024 * 1024,  # 150MB for GitHub issues
    default_ttl=3600,  # 1 hour
    redis_manager=redis_manager
)

# Issue analysis cache for storing analysis results
issue_analysis_cache = EnhancedCacheManager(
    namespace="issue_analysis",
    max_memory_size=200 * 1024 * 1024,  # 200MB for analysis results
    default_ttl=24 * 3600,  # 24 hours (analyses are expensive)
    redis_manager=redis_manager
)

# Compatibility decorators (maintaining existing interface)
def cache_rag_result(ttl: Optional[int] = None):
    """Enhanced RAG caching decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not settings.ENABLE_RAG_CACHING:
                return await func(*args, **kwargs)
            
            cache_key = rag_cache._generate_cache_key(*args, **kwargs)
            
            # Try cache first
            cached = await rag_cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache with intelligent TTL
            cache_ttl = ttl or settings.CACHE_TTL_RAG
            # Longer TTL for expensive RAG operations
            if hasattr(result, '__len__') and len(str(result)) > 10000:
                cache_ttl *= 2  # Double TTL for large contexts
            
            await rag_cache.set(cache_key, result, cache_ttl)
            return result
        return wrapper
    return decorator

def cache_response(ttl: Optional[int] = None):
    """Enhanced LLM response caching decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not settings.ENABLE_RESPONSE_CACHING:
                return await func(*args, **kwargs)
            
            cache_key = response_cache._generate_cache_key(*args, **kwargs)
            
            cached = await response_cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = await func(*args, **kwargs)
            
            cache_ttl = ttl or settings.CACHE_TTL_RESPONSE
            await response_cache.set(cache_key, result, cache_ttl)
            return result
        return wrapper
    return decorator

# Cleanup function
async def cleanup_caches_periodically():
    """Enhanced cleanup with Redis support"""
    while True:
        await asyncio.sleep(60)  # Every minute
        try:
            await rag_cache.clear_expired()
            await response_cache.clear_expired()
            await folder_cache.clear_expired()
            await issue_cache.clear_expired()
            await issue_analysis_cache.clear_expired()
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# Initialize Redis on import
async def initialize_redis_cache():
    """Initialize Redis connection"""
    await redis_manager.initialize() 