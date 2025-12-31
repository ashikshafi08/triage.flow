"""Agent pool for reusing agent instances - extracted from core.py"""
import logging
from typing import Dict, Any
from llama_index.core.agent import ReActAgent
from .prompts import DEFAULT_SYSTEM_PROMPT
from .tool_registry import create_tools_for_subset

logger = logging.getLogger(__name__)

class AgentPool:
    """Agent pool for reusing agent instances to reduce creation overhead."""

    def __init__(self, explorer_instance):
        self.explorer = explorer_instance
        self._agents = {"simple": None, "moderate": None, "complex": None}
        self._cached_tools = {}
        self._stats = {"simple": 0, "moderate": 0, "complex": 0, "cache_hits": 0, "cache_misses": 0}

    def get_agent_for_query(self, query: str, complexity: str, max_iterations: int, tool_subset: str = None) -> ReActAgent:
        """Get optimized agent for query - uses cached agents when possible."""
        # Simple queries with standard config use cached agent
        if complexity == "simple" and max_iterations <= 15 and not tool_subset:
            if not self._agents["simple"]:
                self._agents["simple"] = self._create_agent(15, "search")
            self._agents["simple"].max_iterations = max_iterations
            self._stats["simple"] += 1; self._stats["cache_hits"] += 1
            return self._agents["simple"]

        # Moderate complexity queries
        if complexity == "moderate" and max_iterations <= 25 and not tool_subset:
            if not self._agents["moderate"]:
                self._agents["moderate"] = self._create_agent(25, None)
            self._agents["moderate"].max_iterations = max_iterations
            self._stats["moderate"] += 1; self._stats["cache_hits"] += 1
            return self._agents["moderate"]

        # Complex or custom - create new agent
        self._stats["complex"] += 1; self._stats["cache_misses"] += 1
        return self._create_agent(max_iterations, tool_subset)

    def _create_agent(self, max_iterations: int, tool_subset: str = None) -> ReActAgent:
        """Create optimized agent with optional tool subset."""
        if tool_subset and tool_subset != "comprehensive":
            if tool_subset not in self._cached_tools:
                self._cached_tools[tool_subset] = create_tools_for_subset(self.explorer, tool_subset)
            tools = self._cached_tools[tool_subset]
        else:
            tools = self.explorer.tools

        return ReActAgent.from_tools(tools=tools, llm=self.explorer.base_llm, memory=self.explorer.memory,
                                      verbose=True, max_iterations=max_iterations, system_prompt=DEFAULT_SYSTEM_PROMPT)

    def get_usage_stats(self) -> Dict[str, Any]:
        total = sum(self._stats[k] for k in ["simple", "moderate", "complex"])
        hit_rate = (self._stats["cache_hits"] / total * 100) if total > 0 else 0
        return {**self._stats, "total_requests": total, "cache_hit_rate": f"{hit_rate:.1f}%"}

    def clear_cache(self):
        self._agents = {"simple": None, "moderate": None, "complex": None}
        self._cached_tools = {}
        logger.info("Agent pool cache cleared")
