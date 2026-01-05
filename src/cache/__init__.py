"""
DEPRECATED: This module has been replaced by TypeScript in mastra/src/mastra/context/

These are stub exports to prevent import errors during migration.
Actual implementation is in: mastra/src/mastra/context/contextCache.ts
"""

from typing import Any, Optional


# Stub cache for issue analysis
issue_analysis_cache: Optional[Any] = None


def get_redis_client(*args, **kwargs):
    """DEPRECATED: Redis has been replaced by TTL cache in TypeScript."""
    raise NotImplementedError(
        "Redis cache has been migrated to TypeScript. "
        "Use ContextCache from mastra/src/mastra/context/ instead."
    )
