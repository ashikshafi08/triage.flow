"""
Context-Aware Tool Wrapper

This module wraps existing tools to provide enhanced context sharing and coordination.
It implements the principles from Cognition AI's blog post about avoiding multi-agent
fragmentation by ensuring all tools share context and coordinate decisions.
"""

import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from functools import wraps
from llama_index.core.tools import FunctionTool

from .context_manager import ContextManager, ToolExecution

logger = logging.getLogger(__name__)

class ContextAwareTool:
    """
    Wrapper for tools that provides enhanced context sharing
    """
    
    def __init__(
        self, 
        original_function: Callable,
        tool_name: str,
        description: str,
        context_manager: ContextManager
    ):
        self.original_function = original_function
        self.tool_name = tool_name
        self.description = description
        self.context_manager = context_manager
        self.execution_count = 0
    
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool with enhanced context"""
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # Get context for this tool execution
            context = self.context_manager.get_context_for_tool(self.tool_name, kwargs)
            
            # Check for cached results
            if "cached_result" in context:
                logger.debug(f"Using cached result for {self.tool_name}")
                cached = context["cached_result"]
                # Check if cache is still valid (within 5 minutes for most operations)
                cache_age = time.time() - cached["timestamp"].timestamp()
                if cache_age < 300:  # 5 minutes
                    return cached["result"]
            
            # Store context for internal use but don't pass to original function
            self._current_context = context
            
            # Execute the original function with ONLY the original parameters
            # Do not pass any context parameters to avoid "unexpected keyword argument" errors
            result = self.original_function(*args, **kwargs)
            
            # Post-process result with context
            enhanced_result = self._enhance_result_with_context(result, context)
            
            # Record the execution
            execution_time = time.time() - start_time
            self.context_manager.record_execution(
                tool_name=self.tool_name,
                parameters=kwargs,
                result=enhanced_result,
                execution_time=execution_time,
                context_used=context
            )
            
            return enhanced_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in context-aware execution of {self.tool_name}: {e}")
            
            # Record failed execution
            self.context_manager.record_execution(
                tool_name=self.tool_name,
                parameters=kwargs,
                result=f"Error: {str(e)}",
                execution_time=execution_time,
                context_used={}
            )
            
            # Return original function result as fallback
            return self.original_function(*args, **kwargs)
    
    def _enhance_parameters_with_context(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method is no longer used since we don't modify parameters.
        Context is used internally for result enhancement and logging only.
        """
        # Return original parameters unchanged to avoid parameter conflicts
        return parameters.copy()
    
    def _enhance_result_with_context(self, result: Any, context: Dict[str, Any]) -> Any:
        """Enhance tool result with context information"""
        try:
            # Only enhance results if they are JSON-parseable dictionaries
            # and if the enhancement would be beneficial
            if isinstance(result, str) and self._should_enhance_result():
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict):
                        enhanced_result = self._add_context_to_result(parsed_result, context)
                        return json.dumps(enhanced_result, indent=2)
                except json.JSONDecodeError:
                    # If it's not JSON, just return as-is
                    pass
            
            # If result is already a dict and should be enhanced
            elif isinstance(result, dict) and self._should_enhance_result():
                return self._add_context_to_result(result, context)
            
            # For all other cases, return result unchanged
            return result
            
        except Exception as e:
            logger.warning(f"Error enhancing result for {self.tool_name}: {e}")
            return result
    
    def _should_enhance_result(self) -> bool:
        """Determine if this tool's result should be enhanced with context"""
        # Only enhance results for tools that would benefit from context information
        # and avoid enhancing simple string responses
        enhance_tools = [
            "explore_directory", "read_file", "analyze_file_structure",
            "search_codebase", "find_related_files", "semantic_content_search"
        ]
        return any(tool in self.tool_name for tool in enhance_tools)
    
    def _add_context_to_result(self, result_dict: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Add context information to result dictionary"""
        enhanced = result_dict.copy()
        
        # Only add context metadata if there's meaningful context to add
        context_metadata = {}
        
        if context.get("related_executions"):
            context_metadata["related_executions_count"] = len(context["related_executions"])
        
        if context.get("relevant_files"):
            context_metadata["relevant_files_count"] = len(context["relevant_files"])
            # Add a sample of relevant files (not all to avoid bloat)
            relevant_files = list(context["relevant_files"].keys())[:3]
            if relevant_files:
                context_metadata["sample_relevant_files"] = relevant_files
        
        if context.get("previous_decisions"):
            context_metadata["previous_decisions_count"] = len(context["previous_decisions"])
        
        # Only add metadata if there's something meaningful to add
        if context_metadata:
            enhanced["_context_metadata"] = {
                "tool_name": self.tool_name,
                "execution_count": self.execution_count,
                **context_metadata
            }
        
        return enhanced

class ContextAwareToolFactory:
    """
    Factory for creating context-aware tools
    """
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
    
    def create_context_aware_tools(self, explorer) -> List[FunctionTool]:
        """
        Create context-aware versions of all tools
        """
        tools = []
        
        # File Operations
        tools.extend(self._create_file_operation_tools(explorer))
        
        # Search Operations
        tools.extend(self._create_search_operation_tools(explorer))
        
        # Git Operations
        tools.extend(self._create_git_operation_tools(explorer))
        
        # Issue Operations
        tools.extend(self._create_issue_operation_tools(explorer))
        
        # PR Operations
        tools.extend(self._create_pr_operation_tools(explorer))
        
        # Code Generation Operations
        tools.extend(self._create_code_generation_tools(explorer))
        
        return tools
    
    def _create_file_operation_tools(self, explorer) -> List[FunctionTool]:
        """Create context-aware file operation tools"""
        return [
            self._wrap_tool(
                explorer.file_ops.explore_directory,
                "explore_directory",
                "Explore directory contents with enhanced context from previous explorations. Use directory_path parameter with path relative to repo root (e.g., 'src' or '' for root)."
            ),
            self._wrap_tool(
                explorer.file_ops.read_file,
                "read_file", 
                "Read file contents with context about related files and previous analyses"
            ),
            self._wrap_tool(
                explorer.file_ops.analyze_file_structure,
                "analyze_file_structure",
                "Analyze file structure with context from related files and previous analyses"
            )
        ]
    
    def _create_search_operation_tools(self, explorer) -> List[FunctionTool]:
        """Create context-aware search operation tools"""
        return [
            self._wrap_tool(
                explorer.search_ops.search_codebase,
                "search_codebase",
                "Search codebase with context from previous searches and discovered files. Use query parameter for search term and file_types parameter for extensions list (e.g., ['.py', '.js'])."
            ),
            self._wrap_tool(
                explorer.search_ops.find_related_files,
                "find_related_files",
                "Find related files with context from previous file discoveries and analyses"
            ),
            self._wrap_tool(
                explorer.search_ops.semantic_content_search,
                "semantic_content_search",
                "Semantic search with context from previous searches and file analyses"
            )
        ]
    
    def _create_git_operation_tools(self, explorer) -> List[FunctionTool]:
        """Create context-aware git operation tools"""
        git_tools = [
            ("git_blame_function", "Get git blame with context from related file analyses"),
            ("who_last_edited_line", "Get line edit info with context from file history"),
            ("git_blame_at_commit", "Get historical blame with commit context"),
            ("find_commits_touching_function", "Find function commits with evolution context"),
            ("get_function_evolution", "Get function evolution with comprehensive history context"),
            ("find_pr_closing_commit", "Find PR commits with issue context"),
            ("get_issue_closing_info", "Get issue closing info with PR and commit context"),
            ("get_open_issues_related_to_commit", "Find related issues with commit analysis context"),
            ("find_when_feature_was_added", "Find feature addition with comprehensive git context"),
            ("search_commits", "Search commits with context from previous git analyses"),
            ("get_file_timeline", "Get file timeline with comprehensive history context"),
            ("get_file_commit_statistics", "Get file stats with analysis context"),
            ("get_commit_details", "Get commit details with related commit context"),
            ("analyze_commit_patterns", "Analyze patterns with comprehensive git context"),
            ("get_file_history", "Get file history with issue and PR context"),
            ("summarize_feature_evolution", "Summarize evolution with comprehensive context"),
            ("who_implemented_this", "Find implementer with git history context")
        ]
        
        return [
            self._wrap_tool(
                getattr(explorer.git_ops, tool_name),
                tool_name,
                description
            )
            for tool_name, description in git_tools
        ]
    
    def _create_issue_operation_tools(self, explorer) -> List[FunctionTool]:
        """Create context-aware issue operation tools"""
        issue_tools = [
            ("analyze_github_issue", "Analyze GitHub issue with context from related issues and files. Use issue_identifier parameter with issue number (e.g., '1440') or full URL."),
            ("find_issue_related_files", "Find issue-related files with comprehensive file context. Use issue_description parameter with the issue description text and search_depth parameter ('surface' or 'deep')."),
            ("related_issues", "Find related issues with context from previous issue analyses. Use query parameter with issue title, bug description, or error message to search for similar issues."),
            ("get_issue_closing_info", "Get issue closing info with PR and commit context"),
            ("get_open_issues_related_to_commit", "Find related issues with commit context"),
            ("find_issues_related_to_file", "Find file-related issues with comprehensive context"),
            ("get_issue_resolution_summary", "Get resolution summary with PR and commit context"),
            ("check_issue_status_and_linked_pr", "Check issue status with comprehensive PR context"),
            ("regression_detector", "Detect regressions with historical issue context")
        ]
        
        return [
            self._wrap_tool(
                getattr(explorer.issue_ops, tool_name),
                tool_name,
                description
            )
            for tool_name, description in issue_tools
        ]
    
    def _create_pr_operation_tools(self, explorer) -> List[FunctionTool]:
        """Create context-aware PR operation tools"""
        pr_tools = [
            ("get_pr_for_issue", "Get PR for issue with comprehensive issue context"),
            ("get_pr_diff", "Get PR diff with file and commit context. Use pr_number parameter with the PR number."),
            ("get_files_changed_in_pr", "Get files changed in PR with comprehensive file context. Use pr_number parameter with the PR number."),
            ("get_pr_summary", "Get PR summary with change context. Use pr_number parameter with the PR number."),
            ("find_open_prs_for_issue", "Find open PRs for issue with comprehensive context"),
            ("get_open_pr_status", "Get open PR status with review and CI context"),
            ("find_open_prs_by_files", "Find open PRs by files with file context"),
            ("search_open_prs", "Search open PRs with comprehensive search context"),
            ("check_pr_readiness", "Check PR readiness with review and status context"),
            ("find_feature_introducing_pr", "Find feature introducing PR with historical context"),
            ("get_pr_details_from_github", "Get PR details from GitHub with comprehensive metadata"),
            ("get_pr_analysis", "Get comprehensive PR analysis with combined local and GitHub data")
        ]
        
        tools = []
        for tool_name, description in pr_tools:
            if hasattr(explorer.pr_ops, tool_name):
                tools.append(
                    self._wrap_tool(
                        getattr(explorer.pr_ops, tool_name),
                        tool_name,
                        description
                    )
                )
        
        return tools
    
    def _create_code_generation_tools(self, explorer) -> List[FunctionTool]:
        """Create context-aware code generation tools"""
        return [
            self._wrap_tool(
                explorer.code_gen_ops.generate_code_example,
                "generate_code_example",
                "Generate code examples with context from analyzed files and patterns"
            ),
            self._wrap_tool(
                explorer.code_gen_ops.write_complete_code,
                "write_complete_code",
                "Write complete code with context from file analyses and examples"
            )
        ]
    
    def _wrap_tool(self, original_function: Callable, tool_name: str, description: str) -> FunctionTool:
        """Wrap a function with context awareness"""
        context_aware_tool = ContextAwareTool(
            original_function=original_function,
            tool_name=tool_name,
            description=description,
            context_manager=self.context_manager
        )
        
        return FunctionTool.from_defaults(
            fn=context_aware_tool,
            name=tool_name,
            description=description
        ) 