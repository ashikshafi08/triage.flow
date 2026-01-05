"""
DEPRECATED: This module has been replaced by TypeScript in mastra/src/mastra/rag/

These are stub classes to prevent import errors during migration.
Actual implementation is in: mastra/src/mastra/rag/codebaseRag.ts
"""

from typing import Any, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class IssueContext:
    """Stub for issue context."""
    title: str = ""
    body: str = ""
    labels: List[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []


class LocalRepoContextExtractor:
    """DEPRECATED: Use mastra/src/mastra/rag/codebaseRag.ts instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LocalRepoContextExtractor has been migrated to TypeScript. "
            "Use CodebaseRag from mastra/src/mastra/rag/ instead."
        )


class IssueIndexer:
    """DEPRECATED: Use mastra/src/mastra/rag/codebaseRag.ts instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("IssueIndexer has been migrated to TypeScript.")


class IssueAwareRAG:
    """DEPRECATED: Use mastra/src/mastra/rag/codebaseRag.ts instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "IssueAwareRAG has been migrated to TypeScript. "
            "Use CodebaseRag from mastra/src/mastra/rag/ instead."
        )


class AgenticRAGSystem:
    """DEPRECATED: Use mastra/src/mastra/rag/codebaseRag.ts instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AgenticRAGSystem has been migrated to TypeScript. "
            "Use CodebaseRag from mastra/src/mastra/rag/ instead."
        )
