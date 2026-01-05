"""
DEPRECATED: This module has been replaced by TypeScript in mastra/src/mastra/storage/

These are stub classes to prevent import errors during migration.
Actual implementation is in: mastra/src/mastra/storage/sessionStorage.ts
"""

from typing import Any, Optional, Dict


class SessionManager:
    """DEPRECATED: Use mastra/src/mastra/storage/sessionStorage.ts instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SessionManager has been migrated to TypeScript. "
            "Use SessionStorage from mastra/src/mastra/storage/ instead."
        )


# Stub instance for imports
session_manager: Optional[SessionManager] = None
