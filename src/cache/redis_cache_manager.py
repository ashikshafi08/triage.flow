"""
DEPRECATED: This module has been replaced by TypeScript in mastra/src/mastra/context/

These are stub classes to prevent import errors during migration.
Actual implementation is in: mastra/src/mastra/context/contextCache.ts
"""

from typing import Any, Optional


def get_redis_client(*args, **kwargs):
    """DEPRECATED: Redis has been replaced by TTL cache in TypeScript."""
    raise NotImplementedError(
        "Redis cache has been migrated to TypeScript. "
        "Use ContextCache from mastra/src/mastra/context/ instead."
    )


class RedisCacheManager:
    """DEPRECATED: Use mastra/src/mastra/context/contextCache.ts instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RedisCacheManager has been migrated to TypeScript. "
            "Use ContextCache from mastra/src/mastra/context/ instead."
        )


class EnhancedCacheManager:
    """DEPRECATED: Use mastra/src/mastra/context/contextCache.ts instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "EnhancedCacheManager has been migrated to TypeScript. "
            "Use ContextCache from mastra/src/mastra/context/ instead."
        )
