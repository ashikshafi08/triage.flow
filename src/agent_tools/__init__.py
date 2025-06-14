# src/agent_tools/__init__.py
# This file makes Python treat the directory as a package.

# Import the main class that replaces the monolithic AgenticCodebaseExplorer
from .core import AgenticCodebaseExplorer, capture_output

# Import all operational classes for direct use if needed
from .file_operations import FileOperations
from .search_operations import SearchOperations
from .code_generation import CodeGenerationOperations
from .git_operations import GitOperations
from .issue_operations import IssueOperations
from .pr_operations import PROperations

# Import tool registry for creating tools
from .tool_registry import create_all_tools

# Import context management components
from .context_manager import ContextManager, ToolExecution, ExecutionContext
from .context_aware_tools import ContextAwareTool, ContextAwareToolFactory

# Import utilities
from .utilities import (
    get_current_head_sha,
    extract_repo_info,
    blame_line_cached,
    chunk_large_output,
    get_repo_url_from_path,
    extract_functions,
    extract_classes
)

# Import response handling utilities
from .response_handling import (
    parse_react_steps,
    format_agentic_response,
    clean_captured_output,
    extract_clean_answer,
    basic_cleanup,
    get_natural_exploration_suggestions,
    get_natural_error_recovery
)

# Import LLM configuration
from .llm_config import get_llm_instance

# Import prompts
from .prompts import DEFAULT_SYSTEM_PROMPT, COMMIT_INDEX_SYSTEM_PROMPT

# Make FunctionTool available for compatibility
from llama_index.core.tools import FunctionTool

__all__ = [
    # Main class
    'AgenticCodebaseExplorer',
    'capture_output',
    
    # Operational classes
    'FileOperations',
    'SearchOperations', 
    'CodeGenerationOperations',
    'GitOperations',
    'IssueOperations',
    'PROperations',
    
    # Tool registry
    'create_all_tools',
    
    # Context management
    'ContextManager',
    'ToolExecution',
    'ExecutionContext',
    'ContextAwareTool',
    'ContextAwareToolFactory',
    
    # Utilities
    'get_current_head_sha',
    'extract_repo_info',
    'blame_line_cached',
    'chunk_large_output',
    'get_repo_url_from_path',
    'extract_functions',
    'extract_classes',
    
    # Response handling
    'parse_react_steps',
    'format_agentic_response',
    'clean_captured_output',
    'extract_clean_answer',
    'basic_cleanup',
    'get_natural_exploration_suggestions',
    'get_natural_error_recovery',
    
    # LLM and prompts
    'get_llm_instance',
    'DEFAULT_SYSTEM_PROMPT',
    'COMMIT_INDEX_SYSTEM_PROMPT',
    
    # For compatibility
    'FunctionTool'
]
