"""
Parallel Tool Execution System

This module implements intelligent parallel execution of agent tools to reduce
overall query processing time while maintaining safety and coherence.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Set, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import threading

logger = logging.getLogger(__name__)

@dataclass
class ToolDependency:
    """Represents a dependency relationship between tools"""
    tool_name: str
    depends_on: List[str]
    output_used_by: List[str]
    can_run_parallel: bool = True
    execution_priority: int = 1  # 1-5, higher is more important

@dataclass 
class ExecutionResult:
    """Result of a tool execution"""
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None

class ToolExecutionGraph:
    """
    Manages dependencies and execution order for parallel tool execution.
    """
    
    def __init__(self):
        self.dependencies = {}
        self.tool_characteristics = {}
        self._init_tool_dependencies()
    
    def _init_tool_dependencies(self):
        """Initialize known tool dependencies and characteristics"""
        
        # File operations - can often run in parallel
        self.dependencies.update({
            "read_file": ToolDependency("read_file", [], ["analyze_file_structure"], True, 2),
            "explore_directory": ToolDependency("explore_directory", [], ["read_file"], True, 1),
            "analyze_file_structure": ToolDependency("analyze_file_structure", ["read_file"], [], True, 2),
        })
        
        # Search operations - independent and parallelizable
        self.dependencies.update({
            "search_codebase": ToolDependency("search_codebase", [], [], True, 3),
            "find_related_files": ToolDependency("find_related_files", [], [], True, 2),
            "semantic_content_search": ToolDependency("semantic_content_search", [], [], True, 3),
        })
        
        # Git operations - some dependencies exist
        self.dependencies.update({
            "git_blame_function": ToolDependency("git_blame_function", [], [], True, 2),
            "get_file_history": ToolDependency("get_file_history", [], [], True, 2),
            "get_commit_details": ToolDependency("get_commit_details", [], [], True, 1),
            "search_commits": ToolDependency("search_commits", [], [], True, 2),
            "who_implemented_this": ToolDependency("who_implemented_this", [], [], True, 2),
        })
        
        # Issue operations - may depend on file/git operations
        self.dependencies.update({
            "analyze_github_issue": ToolDependency("analyze_github_issue", [], ["find_issue_related_files"], True, 3),
            "find_issue_related_files": ToolDependency("find_issue_related_files", ["analyze_github_issue"], [], True, 2),
            "related_issues": ToolDependency("related_issues", [], [], True, 2),
        })
        
        # PR operations - similar to issue operations
        self.dependencies.update({
            "get_pr_diff": ToolDependency("get_pr_diff", [], ["get_files_changed_in_pr"], True, 2),
            "get_files_changed_in_pr": ToolDependency("get_files_changed_in_pr", ["get_pr_diff"], [], True, 1),
            "get_pr_summary": ToolDependency("get_pr_summary", ["get_pr_diff"], [], True, 2),
        })
        
        # Code generation - typically depends on analysis
        self.dependencies.update({
            "generate_code_example": ToolDependency("generate_code_example", ["read_file", "search_codebase"], [], False, 1),
            "write_complete_code": ToolDependency("write_complete_code", ["read_file", "analyze_file_structure"], [], False, 1),
        })
    
    def can_execute_parallel(self, tool_names: List[str]) -> List[List[str]]:
        \"\"\"
        Determine which tools can be executed in parallel batches.
        Returns list of batches, where each batch can run in parallel.
        \"\"\"\n        if not tool_names:\n            return []\n        \n        # Build dependency graph\n        remaining_tools = set(tool_names)\n        execution_batches = []\n        \n        while remaining_tools:\n            # Find tools that have no unmet dependencies\n            ready_tools = []\n            for tool in remaining_tools:\n                deps = self.dependencies.get(tool, ToolDependency(tool, [], [], True))\n                unmet_deps = set(deps.depends_on) & remaining_tools\n                \n                if not unmet_deps and deps.can_run_parallel:\n                    ready_tools.append(tool)\n            \n            # If no tools are ready, break dependency cycle by picking highest priority\n            if not ready_tools:\n                priority_tool = max(remaining_tools, \n                                  key=lambda t: self.dependencies.get(t, ToolDependency(t, [], [], True, 1)).execution_priority)\n                ready_tools = [priority_tool]\n                logger.warning(f\"Breaking dependency cycle by prioritizing {priority_tool}\")\n            \n            # Add this batch and remove from remaining\n            execution_batches.append(ready_tools)\n            remaining_tools -= set(ready_tools)\n        \n        return execution_batches\n    \n    def get_safe_parallel_groups(self, tool_names: List[str]) -> List[List[str]]:\n        \"\"\"Get safe parallel execution groups with conservative dependencies\"\"\"\n        # Conservative groupings for safety\n        safe_groups = {\n            \"read_operations\": [\"read_file\", \"explore_directory\"],\n            \"search_operations\": [\"search_codebase\", \"find_related_files\", \"semantic_content_search\"],\n            \"git_readonly\": [\"git_blame_function\", \"get_file_history\", \"get_commit_details\", \"search_commits\"],\n            \"issue_readonly\": [\"analyze_github_issue\", \"related_issues\"],\n            \"pr_readonly\": [\"get_pr_diff\", \"get_files_changed_in_pr\", \"get_pr_summary\"]\n        }\n        \n        groups = []\n        used_tools = set()\n        \n        for group_name, group_tools in safe_groups.items():\n            group_intersection = [t for t in tool_names if t in group_tools and t not in used_tools]\n            if group_intersection:\n                groups.append(group_intersection)\n                used_tools.update(group_intersection)\n        \n        # Add remaining tools as individual groups\n        for tool in tool_names:\n            if tool not in used_tools:\n                groups.append([tool])\n        \n        return groups

class ParallelToolExecutor:
    \"\"\"
    Executes tools in parallel when safe to do so.
    \"\"\"
    
    def __init__(self, max_workers: int = 4):\n        self.max_workers = max_workers\n        self.execution_graph = ToolExecutionGraph()\n        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)\n        self.execution_stats = {\n            \"total_executions\": 0,\n            \"parallel_executions\": 0,\n            \"time_saved\": 0.0,\n            \"errors\": 0\n        }\n    \n    async def execute_tools_parallel(self, tool_calls: List[Tuple[Callable, List, Dict]], timeout: float = 30.0) -> List[ExecutionResult]:\n        \"\"\"Execute a list of tool calls in parallel when possible\"\"\"\n        if not tool_calls:\n            return []\n        \n        if len(tool_calls) == 1:\n            # Single tool, execute directly\n            return [await self._execute_single_tool(tool_calls[0])]\n        \n        # Extract tool names for dependency analysis\n        tool_names = [self._extract_tool_name(call[0]) for call in tool_calls]\n        \n        # Get parallel execution groups\n        execution_groups = self.execution_graph.get_safe_parallel_groups(tool_names)\n        \n        logger.info(f\"Executing {len(tool_calls)} tools in {len(execution_groups)} parallel groups\")\n        \n        all_results = []\n        start_time = time.time()\n        \n        for group_idx, tool_group in enumerate(execution_groups):\n            # Find tool calls for this group\n            group_calls = [call for call in tool_calls if self._extract_tool_name(call[0]) in tool_group]\n            \n            if len(group_calls) == 1:\n                # Single tool in group\n                result = await self._execute_single_tool(group_calls[0])\n                all_results.append(result)\n            else:\n                # Parallel execution within group\n                group_results = await self._execute_group_parallel(group_calls, timeout)\n                all_results.extend(group_results)\n                self.execution_stats[\"parallel_executions\"] += len(group_calls)\n            \n            self.execution_stats[\"total_executions\"] += len(group_calls)\n        \n        total_time = time.time() - start_time\n        logger.info(f\"Completed parallel tool execution in {total_time:.2f}s\")\n        \n        return all_results\n    \n    async def _execute_group_parallel(self, tool_calls: List[Tuple[Callable, List, Dict]], timeout: float) -> List[ExecutionResult]:\n        \"\"\"Execute a group of tools in parallel\"\"\"\n        tasks = []\n        for tool_call in tool_calls:\n            task = asyncio.create_task(self._execute_single_tool(tool_call))\n            tasks.append(task)\n        \n        try:\n            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)\n            \n            # Process results and handle exceptions\n            processed_results = []\n            for i, result in enumerate(results):\n                if isinstance(result, Exception):\n                    tool_name = self._extract_tool_name(tool_calls[i][0])\n                    processed_results.append(ExecutionResult(\n                        tool_name=tool_name,\n                        success=False,\n                        result=None,\n                        execution_time=0.0,\n                        error=str(result)\n                    ))\n                    self.execution_stats[\"errors\"] += 1\n                else:\n                    processed_results.append(result)\n            \n            return processed_results\n            \n        except asyncio.TimeoutError:\n            logger.error(f\"Parallel execution timed out after {timeout}s\")\n            # Return timeout errors for all tools\n            return [ExecutionResult(\n                tool_name=self._extract_tool_name(call[0]),\n                success=False,\n                result=None,\n                execution_time=timeout,\n                error=\"Execution timeout\"\n            ) for call in tool_calls]\n    \n    async def _execute_single_tool(self, tool_call: Tuple[Callable, List, Dict]) -> ExecutionResult:\n        \"\"\"Execute a single tool and return the result\"\"\"\n        func, args, kwargs = tool_call\n        tool_name = self._extract_tool_name(func)\n        \n        start_time = time.time()\n        try:\n            # Handle both sync and async functions\n            if asyncio.iscoroutinefunction(func):\n                result = await func(*args, **kwargs)\n            else:\n                # Run sync function in thread pool to avoid blocking\n                loop = asyncio.get_event_loop()\n                result = await loop.run_in_executor(self.thread_pool, lambda: func(*args, **kwargs))\n            \n            execution_time = time.time() - start_time\n            return ExecutionResult(\n                tool_name=tool_name,\n                success=True,\n                result=result,\n                execution_time=execution_time\n            )\n            \n        except Exception as e:\n            execution_time = time.time() - start_time\n            logger.error(f\"Tool {tool_name} failed: {e}\")\n            return ExecutionResult(\n                tool_name=tool_name,\n                success=False,\n                result=None,\n                execution_time=execution_time,\n                error=str(e)\n            )\n    \n    def _extract_tool_name(self, func: Callable) -> str:\n        \"\"\"Extract tool name from function\"\"\"\n        if hasattr(func, '__name__'):\n            return func.__name__\n        elif hasattr(func, 'metadata') and hasattr(func.metadata, 'name'):\n            return func.metadata.name\n        else:\n            return str(func)\n    \n    def get_execution_stats(self) -> Dict[str, Any]:\n        \"\"\"Get execution statistics\"\"\"\n        total = self.execution_stats[\"total_executions\"]\n        parallel = self.execution_stats[\"parallel_executions\"]\n        parallel_rate = (parallel / total * 100) if total > 0 else 0\n        \n        return {\n            **self.execution_stats,\n            \"parallel_execution_rate\": f\"{parallel_rate:.1f}%\"\n        }\n    \n    def reset_stats(self):\n        \"\"\"Reset execution statistics\"\"\"\n        self.execution_stats = {\n            \"total_executions\": 0,\n            \"parallel_executions\": 0,\n            \"time_saved\": 0.0,\n            \"errors\": 0\n        }\n    \n    def cleanup(self):\n        \"\"\"Cleanup resources\"\"\"\n        self.thread_pool.shutdown(wait=True)\n\ndef parallel_tool_execution(max_workers: int = 4):\n    \"\"\"Decorator for enabling parallel tool execution in agent methods\"\"\"\n    def decorator(func):\n        @wraps(func)\n        async def wrapper(*args, **kwargs):\n            # This would be integrated into the agent's tool execution logic\n            # For now, just call the original function\n            return await func(*args, **kwargs)\n        return wrapper\n    return decorator\n\n# Global executor instance for reuse\n_global_executor = None\n\ndef get_parallel_executor(max_workers: int = 4) -> ParallelToolExecutor:\n    \"\"\"Get or create the global parallel executor\"\"\"\n    global _global_executor\n    if _global_executor is None:\n        _global_executor = ParallelToolExecutor(max_workers)\n    return _global_executor\n\ndef optimize_tool_calls_for_parallel_execution(tool_calls: List[Any]) -> List[List[Any]]:\n    \"\"\"Optimize a list of tool calls for parallel execution\"\"\"\n    executor = get_parallel_executor()\n    \n    # Extract tool names\n    tool_names = []\n    for call in tool_calls:\n        if hasattr(call, 'tool_name'):\n            tool_names.append(call.tool_name)\n        elif hasattr(call, '__name__'):\n            tool_names.append(call.__name__)\n        else:\n            tool_names.append(\"unknown_tool\")\n    \n    # Get parallel execution groups\n    groups = executor.execution_graph.get_safe_parallel_groups(tool_names)\n    \n    # Organize tool calls into groups\n    organized_calls = []\n    for group in groups:\n        group_calls = []\n        for i, tool_name in enumerate(tool_names):\n            if tool_name in group:\n                group_calls.append(tool_calls[i])\n        if group_calls:\n            organized_calls.append(group_calls)\n    \n    return organized_calls