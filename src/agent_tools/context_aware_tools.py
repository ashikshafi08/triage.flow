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

# Parameter aliases to handle mismatched parameter names from agents
PARAMETER_ALIASES = {
    'explore_directory': {
        'dir_path': 'directory_path',
        'path': 'directory_path',
        'args': 'directory_path'  # Handle when agent passes [''] as args
    },
    'find_related_files': {
        'file_path': 'file_path_str',
        'path': 'file_path_str'
    },
    'read_file': {
        'path': 'file_path',
        'file': 'file_path'
    },
    'search_codebase': {
        'search_query': 'query',
        'search_terms': 'query'
    }
}

# Global memoization cache for tool results
_TOOL_RESULT_CACHE = {}

def clear_tool_cache():
    """Clear the global tool result cache"""
    global _TOOL_RESULT_CACHE
    _TOOL_RESULT_CACHE.clear()
    logger.info("Cleared tool result cache")

def get_cache_stats():
    """Get cache statistics"""
    if not _TOOL_RESULT_CACHE:
        return {"size": 0, "oldest": None, "newest": None}
    
    timestamps = [entry['timestamp'] for entry in _TOOL_RESULT_CACHE.values()]
    return {
        "size": len(_TOOL_RESULT_CACHE),
        "oldest": min(timestamps),
        "newest": max(timestamps)
    }

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
            # Normalize parameters using aliases
            normalized_kwargs = self._normalize_parameters(args, kwargs)
            
            # Create cache key for memoization
            cache_key = self._create_cache_key(args, normalized_kwargs)
            
            # Check global memoization cache first
            if cache_key in _TOOL_RESULT_CACHE:
                cache_entry = _TOOL_RESULT_CACHE[cache_key]
                cache_age = time.time() - cache_entry['timestamp']
                if cache_age < 300:  # 5 minutes
                    logger.info(f"Using memoized result for {self.tool_name} (age: {cache_age:.1f}s)")
                    return cache_entry['result']
            
            # Get context for this tool execution
            context = self.context_manager.get_context_for_tool(self.tool_name, normalized_kwargs)
            
            # Check for cached results from context
            if "cached_result" in context:
                logger.debug(f"Using context cached result for {self.tool_name}")
                cached = context["cached_result"]
                # Check if cache is still valid (within 5 minutes for most operations)
                cache_age = time.time() - cached["timestamp"].timestamp()
                if cache_age < 300:  # 5 minutes
                    return cached["result"]
            
            # Store context for internal use but don't pass to original function
            self._current_context = context
            
            # Execute the original function with ONLY the original parameters
            # Do not pass any context parameters to avoid "unexpected keyword argument" errors
            try:
                # Use normalized parameters for execution
                result = self._execute_with_normalized_params(args, normalized_kwargs)
            except TypeError as type_error:
                # If we get a TypeError about unexpected keyword arguments, 
                # filter the parameters and retry immediately
                if "unexpected keyword argument" in str(type_error):
                    logger.warning(f"Got unexpected keyword argument error for {self.tool_name}, filtering parameters")
                    
                    # Handle LlamaIndex format during error recovery
                    if len(args) == 0 and 'args' in kwargs and 'kwargs' in kwargs:
                        llamaindex_args = kwargs.get('args', [])
                        llamaindex_kwargs = kwargs.get('kwargs', {})
                        
                        # Handle AttributedDict or other dict-like objects in error recovery
                        if hasattr(llamaindex_kwargs, '__dict__'):
                            llamaindex_kwargs = dict(llamaindex_kwargs)
                        elif hasattr(llamaindex_kwargs, 'items'):
                            llamaindex_kwargs = dict(llamaindex_kwargs.items())
                        elif isinstance(llamaindex_kwargs, (list, tuple)):
                            try:
                                llamaindex_kwargs = dict(llamaindex_kwargs)
                            except (ValueError, TypeError):
                                logger.warning(f"Failed to convert llamaindex_kwargs to dict in error recovery: {llamaindex_kwargs}")
                                llamaindex_kwargs = {}
                        
                        # Handle args that might be AttributedList or similar
                        if hasattr(llamaindex_args, '__iter__') and not isinstance(llamaindex_args, (str, bytes)):
                            try:
                                llamaindex_args = list(llamaindex_args)
                            except (ValueError, TypeError):
                                logger.warning(f"Failed to convert llamaindex_args to list: {llamaindex_args}")
                                llamaindex_args = []
                        
                        # Filter only the llamaindex_kwargs
                        cleaned_kwargs = self._filter_invalid_parameters(llamaindex_kwargs)
                        logger.info(f"Retrying {self.tool_name} with LlamaIndex format and cleaned parameters: args={llamaindex_args}, kwargs={cleaned_kwargs}")
                        result = self.original_function(*llamaindex_args, **cleaned_kwargs)
                    else:
                        # Standard format filtering
                        cleaned_kwargs = self._filter_invalid_parameters(kwargs)
                        result = self.original_function(*args, **cleaned_kwargs)
                else:
                    raise type_error
            
            # Post-process result with context
            enhanced_result = self._enhance_result_with_context(result, context)
            
            # Store in memoization cache
            _TOOL_RESULT_CACHE[cache_key] = {
                'result': enhanced_result,
                'timestamp': time.time()
            }
            
            # Limit cache size to prevent memory bloat
            if len(_TOOL_RESULT_CACHE) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(_TOOL_RESULT_CACHE.keys(), 
                                   key=lambda k: _TOOL_RESULT_CACHE[k]['timestamp'])[:100]
                for old_key in oldest_keys:
                    del _TOOL_RESULT_CACHE[old_key]
            
            # Record the execution
            execution_time = time.time() - start_time
            self.context_manager.record_execution(
                tool_name=self.tool_name,
                parameters=normalized_kwargs,
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
            
            # Special handling for AttributedDict initialization errors
            error_str = str(e)
            if "AttributedDict.__init__()" in error_str and "takes 1 positional argument but" in error_str:
                logger.warning(f"AttributedDict initialization error detected, attempting parameter normalization for {self.tool_name}")
                try:
                    # Re-normalize parameters with extra AttributedDict handling
                    fixed_kwargs = self._fix_attributed_dict_params(kwargs)
                    return self.original_function(**fixed_kwargs)
                except Exception as fix_error:
                    logger.error(f"AttributedDict fix also failed for {self.tool_name}: {fix_error}")
                    return f"Error: AttributedDict parameter error - {str(e)}"
            
            # Check if error is due to invalid parameters
            elif "unexpected keyword argument" in error_str:
                # Filter out invalid parameters and retry
                cleaned_kwargs = self._filter_invalid_parameters(kwargs)
                logger.info(f"Retrying {self.tool_name} with cleaned parameters: {cleaned_kwargs}")
                try:
                    return self.original_function(*args, **cleaned_kwargs)
                except Exception as retry_error:
                    logger.error(f"Retry also failed for {self.tool_name}: {retry_error}")
                    return f"Error: Tool execution failed - {str(e)}"
            
            # For other errors, return error message instead of retrying with same params
            return f"Error: {str(e)}"
    
    def _normalize_parameters(self, args: tuple, kwargs: dict) -> dict:
        """Normalize parameters using aliases and handle special cases"""
        normalized_kwargs = kwargs.copy()
        
        # Handle LlamaIndex calling convention
        if len(args) == 0 and 'args' in kwargs and 'kwargs' in kwargs:
            llamaindex_args = kwargs.get('args', [])
            llamaindex_kwargs = kwargs.get('kwargs', {})
            logger.debug(f"Detected LlamaIndex calling convention: args={llamaindex_args}, kwargs={llamaindex_kwargs}")
            
            # Handle AttributedDict or other dict-like objects by converting to regular dict
            if hasattr(llamaindex_kwargs, '__dict__'):
                # If it's an object with attributes, convert to dict
                normalized_kwargs = dict(llamaindex_kwargs)
            elif hasattr(llamaindex_kwargs, 'items'):
                # If it's a dict-like object with items() method
                normalized_kwargs = dict(llamaindex_kwargs.items()) 
            elif isinstance(llamaindex_kwargs, (list, tuple)):
                # If it's a list/tuple of key-value pairs (like AttributedDict constructor might receive)
                try:
                    normalized_kwargs = dict(llamaindex_kwargs)
                except (ValueError, TypeError):
                    logger.warning(f"Failed to convert llamaindex_kwargs to dict: {llamaindex_kwargs}")
                    normalized_kwargs = {}
            else:
                # For normal dict, just copy
                normalized_kwargs = llamaindex_kwargs.copy() if llamaindex_kwargs else {}
            
            # Handle cases where args contain the actual parameter
            if llamaindex_args and self.tool_name == 'explore_directory':
                normalized_kwargs['directory_path'] = llamaindex_args[0] if llamaindex_args[0] else ''
            elif llamaindex_args and self.tool_name == 'read_file':
                normalized_kwargs['file_path'] = llamaindex_args[0]
                
        # Handle single positional argument case
        elif len(args) == 1 and isinstance(args[0], str) and not kwargs:
            if self.tool_name == 'explore_directory':
                normalized_kwargs = {'directory_path': args[0]}
            elif self.tool_name == 'read_file':
                normalized_kwargs = {'file_path': args[0]}
            else:
                # For other tools, use the first positional arg as the main parameter
                import inspect
                sig = inspect.signature(self.original_function)
                param_names = list(sig.parameters.keys())
                if param_names and param_names[0] != 'self':
                    normalized_kwargs = {param_names[0]: args[0]}
        
        # Apply parameter aliases
        if self.tool_name in PARAMETER_ALIASES:
            aliases = PARAMETER_ALIASES[self.tool_name]
            for old_name, new_name in aliases.items():
                if old_name in normalized_kwargs:
                    normalized_kwargs[new_name] = normalized_kwargs.pop(old_name)
        
        # Filter out context metadata and other internal parameters
        internal_params = {'_current_context', '_context_metadata', 'args', 'kwargs', 'tool_input', 'input'}
        normalized_kwargs = {k: v for k, v in normalized_kwargs.items() if k not in internal_params}
        
        return normalized_kwargs
    
    def _create_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Create a cache key from arguments"""
        # Create a stable key from the tool name and normalized parameters
        import hashlib
        import json
        
        try:
            # Sort kwargs for consistent key generation
            sorted_kwargs = sorted(kwargs.items()) if kwargs else []
            key_data = {
                'tool': self.tool_name,
                'args': args,
                'kwargs': sorted_kwargs
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception:
            # Fallback to simple string key if JSON serialization fails
            return f"{self.tool_name}_{str(args)}_{str(sorted(kwargs.items()) if kwargs else [])}"
    
    def _execute_with_normalized_params(self, args: tuple, kwargs: dict) -> Any:
        """Execute the original function with normalized parameters"""
        try:
            # Try direct execution with normalized kwargs
            if args:
                return self.original_function(*args, **kwargs)
            else:
                return self.original_function(**kwargs)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                # Filter parameters that the function doesn't accept
                filtered_kwargs = self._filter_invalid_parameters(kwargs)
                logger.warning(f"Filtered invalid parameters for {self.tool_name}: {set(kwargs.keys()) - set(filtered_kwargs.keys())}")
                
                if args:
                    return self.original_function(*args, **filtered_kwargs)
                else:
                    return self.original_function(**filtered_kwargs)
            else:
                raise
    
    def _filter_invalid_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out invalid parameters based on function signature"""
        import inspect
        
        try:
            # Get the function signature
            sig = inspect.signature(self.original_function)
            valid_params = set(sig.parameters.keys())
            
            # Remove 'self' parameter if it exists since it's handled automatically
            valid_params.discard('self')
            
            # Common invalid parameters to filter out
            invalid_params_to_remove = {'args', 'tool_input', 'input', 'kwargs', '_current_context', '_context_metadata'}
            
            # Filter kwargs to only include valid parameters
            cleaned_kwargs = {
                k: v for k, v in kwargs.items() 
                if k in valid_params and k not in invalid_params_to_remove
            }
            
            invalid_params = set(kwargs.keys()) - valid_params - invalid_params_to_remove
            filtered_params = set(kwargs.keys()) & invalid_params_to_remove
            
            if invalid_params or filtered_params:
                logger.warning(f"Filtered invalid parameters for {self.tool_name}: invalid={invalid_params}, filtered={filtered_params}")
            
            return cleaned_kwargs
        except Exception as e:
            logger.error(f"Error filtering parameters for {self.tool_name}: {e}")
            return {}
    
    def _enhance_parameters_with_context(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method is no longer used since we don't modify parameters.
        Context is used internally for result enhancement and logging only.
        """
        # Return original parameters unchanged to avoid parameter conflicts
        return parameters.copy()
    
    def _fix_attributed_dict_params(self, kwargs: dict) -> dict:
        """
        Fix AttributedDict initialization errors by properly extracting parameters.
        This handles cases where LlamaIndex passes AttributedDict objects incorrectly.
        """
        try:
            # Check if we have the LlamaIndex format with 'kwargs' key
            if 'kwargs' in kwargs:
                llamaindex_kwargs = kwargs.get('kwargs', {})
                
                # If it's some kind of AttributedDict-like object that failed to initialize
                if hasattr(llamaindex_kwargs, '__class__') and 'AttributedDict' in str(llamaindex_kwargs.__class__):
                    # Try to extract the underlying data
                    if hasattr(llamaindex_kwargs, '_data'):
                        return llamaindex_kwargs._data
                    elif hasattr(llamaindex_kwargs, '__dict__'):
                        return llamaindex_kwargs.__dict__
                    else:
                        logger.warning(f"AttributedDict object has no accessible data: {llamaindex_kwargs}")
                        return {}
                
                # If it's a list of tuples (which might have been passed to AttributedDict constructor)
                elif isinstance(llamaindex_kwargs, (list, tuple)):
                    try:
                        return dict(llamaindex_kwargs)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert list/tuple to dict: {e}")
                        return {}
                
                # If it's already a dict, just return it
                elif isinstance(llamaindex_kwargs, dict):
                    return llamaindex_kwargs
                
                # If it has items() method, convert to dict
                elif hasattr(llamaindex_kwargs, 'items'):
                    return dict(llamaindex_kwargs.items())
                
                else:
                    logger.warning(f"Unknown kwargs type: {type(llamaindex_kwargs)}")
                    return {}
            
            # If no special kwargs key, try to clean up the whole kwargs dict
            else:
                return self._normalize_parameters((), kwargs)
                
        except Exception as e:
            logger.error(f"Error in _fix_attributed_dict_params: {e}")
            return {}
    
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
                "Search codebase with context from previous searches and discovered files. Use query parameter for search term, file_types parameter for extensions list (e.g., ['.py', '.js']), and optional directory_path parameter to limit search scope (e.g., 'src', 'examples')."
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
            ("search_commits", "Search commits with context from previous git analyses. Use parameters: query (required), k (optional), author_filter (optional), file_filter (optional), path (optional). DO NOT use 'search_by' parameter."),
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