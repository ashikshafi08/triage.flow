"""
DEPRECATED: This module has been replaced by TypeScript in mastra/src/mastra/context/

These are stub classes to prevent import errors during migration.
Actual implementation is in: mastra/src/mastra/context/contextManager.ts
"""

from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolExecution:
    """Stub for tool execution record."""
    id: str = ""
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionContext:
    """Stub for execution context."""
    session_id: str = ""
    query: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    discovered_files: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[ToolExecution] = field(default_factory=list)


class ContextManager:
    """DEPRECATED: Use mastra/src/mastra/context/contextManager.ts instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ContextManager has been migrated to TypeScript. "
            "Use ContextManager from mastra/src/mastra/context/ instead."
        )
