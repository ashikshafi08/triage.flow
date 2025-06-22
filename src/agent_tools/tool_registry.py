# src/agent_tools/tool_registry.py

from typing import List, TYPE_CHECKING, Dict, Set
from llama_index.core.tools import FunctionTool
import asyncio
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .core import AgenticCodebaseExplorer # To access ops instances

def create_async_tool_wrapper(async_func, func_name: str):
    """Create a synchronous wrapper for async functions to be used in tools"""
    def sync_wrapper(*args, **kwargs):
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, but we need to run the coroutine
                # Use asyncio.create_task to properly handle it
                import concurrent.futures
                import threading
                
                # Create a new event loop in a separate thread
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(async_func(*args, **kwargs))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=60)  # 60 second timeout
                    
            except RuntimeError:
                # No event loop running, we can use asyncio.run
                return asyncio.run(async_func(*args, **kwargs))
                
        except Exception as e:
            logger.error(f"Error in async tool wrapper for {func_name}: {e}")
            import json
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})
    
    # Copy function metadata
    sync_wrapper.__name__ = f"{func_name}_sync_wrapper"
    sync_wrapper.__doc__ = getattr(async_func, '__doc__', '')
    
    return sync_wrapper

# Tool subset definitions for performance optimization
TOOL_SUBSETS = {
    "search": {
        "tools": ["search_codebase", "find_related_files", "semantic_content_search", "read_file"],
        "description": "Tools for searching and finding content in the codebase"
    },
    "file": {
        "tools": ["read_file", "explore_directory", "analyze_file_structure", "search_codebase"],
        "description": "Tools for file exploration and analysis"
    },
    "git": {
        "tools": ["git_blame_function", "get_file_history", "get_commit_details", "search_commits", 
                  "find_commits_touching_function", "get_function_evolution", "who_implemented_this", "read_file"],
        "description": "Tools for git history and blame analysis"
    },
    "issue": {
        "tools": ["analyze_github_issue", "find_issue_related_files", "related_issues", 
                  "get_issue_closing_info", "find_issues_related_to_file", "check_issue_status_and_linked_pr", "read_file"],
        "description": "Tools for GitHub issue analysis and tracking"
    },
    "pr": {
        "tools": ["get_pr_for_issue", "get_pr_diff", "get_files_changed_in_pr", "get_pr_summary", 
                  "get_pr_analysis", "find_open_prs_for_issue", "get_open_pr_status", "read_file"],
        "description": "Tools for pull request analysis and tracking"
    },
    "code_gen": {
        "tools": ["generate_code_example", "write_complete_code", "read_file", "search_codebase", "analyze_file_structure"],
        "description": "Tools for code generation and examples"
    },
    "comprehensive": {
        "tools": [],  # Will include all tools
        "description": "All available tools for complex queries"
    }
}

# Essential tools that should be included in most subsets
ESSENTIAL_TOOLS = ["read_file", "search_codebase"]

def create_all_tools(explorer: 'AgenticCodebaseExplorer') -> List[FunctionTool]:
    """
    Creates and returns a list of all FunctionTool instances for the agent.
    It uses the operational classes (file_ops, search_ops, etc.) from the explorer instance.
    """
    tools = [
        # FileOperations
        FunctionTool.from_defaults(
            fn=explorer.file_ops.explore_directory, 
            name="explore_directory",
            description="Explore the contents of a directory in the repository to understand its structure and files. Use directory_path parameter with path relative to repo root (e.g., 'src' or '' for root)."
        ),
        FunctionTool.from_defaults(
            fn=explorer.file_ops.read_file, 
            name="read_file",
            description="Read the complete contents of a specific file in the repository"
        ),
        FunctionTool.from_defaults(
            fn=explorer.file_ops.analyze_file_structure, 
            name="analyze_file_structure",
            description="Analyze the structure and components of a file or directory"
        ),
        # stream_large_file is an async generator, not directly a tool for ReAct agent.
        # It would be used differently if needed by a streaming component.

        # SearchOperations
        FunctionTool.from_defaults(
            fn=explorer.search_ops.search_codebase, 
            name="search_codebase",
            description="Search through the codebase for specific patterns, functions, classes, or concepts. Use query parameter for search term and file_types parameter for extensions list (e.g., ['.py', '.js'])."
        ),
        FunctionTool.from_defaults(
            fn=explorer.search_ops.find_related_files, 
            name="find_related_files",
            description="Find files related to a specific file based on imports, references, and patterns"
        ),
        FunctionTool.from_defaults(
            fn=explorer.search_ops.semantic_content_search, 
            name="semantic_content_search",
            description="Search for content semantically across all files in the repository"
        ),

        # CodeGenerationOperations
        FunctionTool.from_defaults(
            fn=explorer.code_gen_ops.generate_code_example, 
            name="generate_code_example",
            description="Generate code examples based on repository patterns and context"
        ),
        FunctionTool.from_defaults(
            fn=explorer.code_gen_ops.write_complete_code, 
            name="write_complete_code",
            description="Write complete, untruncated code files based on specifications and reference files"
        ),

        # GitOperations (delegates to GitBlameTools, GitHistoryTools, CommitIndexManager)
        FunctionTool.from_defaults(
            fn=explorer.git_ops.git_blame_function, 
            name="git_blame_function",
            description="Get git blame information for a specific function or class, showing who last edited each line"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.who_last_edited_line, 
            name="who_last_edited_line",
            description="Get information about who last edited a specific line in a file"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.git_blame_at_commit, 
            name="git_blame_at_commit",
            description="Get git blame information for a file at a specific commit in history (not current state)"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.find_commits_touching_function, 
            name="find_commits_touching_function",
            description="Find all commits that modified a specific function in a file over time"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.get_function_evolution, 
            name="get_function_evolution",
            description="Get the evolution of a function over time with diff details between commits"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.find_pr_closing_commit, 
            name="find_pr_closing_commit",
            description="Get the merge commit information for a specific PR"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.get_issue_closing_info,
            name="get_issue_closing_info", 
            description="Get detailed information about who closed an issue and with what commit/PR"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.get_open_issues_related_to_commit,
            name="get_open_issues_related_to_commit",
            description="Find open issues that might be related to changes in a specific commit"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.find_when_feature_was_added, 
            name="find_when_feature_was_added",
            description="Find when a specific feature, function, or code pattern was first added to the codebase using git history"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.search_commits, 
            name="search_commits",
            description="Semantic search over commit messages and metadata for deeper insights into code history"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.get_file_timeline, 
            name="get_file_timeline",
            description="Get complete timeline of all commits that touched a specific file (more granular than PR history)"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.get_file_commit_statistics, 
            name="get_file_commit_statistics",
            description="Get comprehensive statistics about file changes and contributors"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.get_commit_details, 
            name="get_commit_details",
            description="Get detailed information about any specific commit by SHA"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.analyze_commit_patterns, 
            name="analyze_commit_patterns",
            description="Find patterns in commit messages, authors, and file changes across repository history"
        ),
        # Additional Git History Tools
        FunctionTool.from_defaults(
            fn=explorer.git_ops.get_file_history,
            name="get_file_history",
            description="Get the timeline of issues/PRs that touched a file. Use this to understand how a file evolved over time."
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.summarize_feature_evolution,
            name="summarize_feature_evolution",
            description="Summarize how a feature evolved over time by finding all related issues, PRs, and changes"
        ),
        FunctionTool.from_defaults(
            fn=explorer.git_ops.who_implemented_this,
            name="who_implemented_this",
            description="Find who initially implemented a class, function, or feature using git history"
        ),

        # IssueOperations (delegates to IssueClosingTools, IssueAwareRAG, SearchOps)
        FunctionTool.from_defaults(
            fn=explorer.issue_ops.analyze_github_issue, 
            name="analyze_github_issue",
            description="Analyze a GitHub issue to understand the problem and provide solution approaches. Use issue_identifier parameter with issue number (e.g., '1440') or full URL."
        ),
        FunctionTool.from_defaults(
            fn=explorer.issue_ops.find_issue_related_files, 
            name="find_issue_related_files",
            description="Find files in the codebase that are related to a specific issue or problem description. Use issue_description parameter with the issue description text and search_depth parameter ('surface' or 'deep')."
        ),
        FunctionTool.from_defaults(
            fn=explorer.issue_ops.related_issues, 
            name="related_issues",
            description="Find similar past GitHub issues in the current repository that might provide context or solutions. Use query parameter with issue title, bug description, or error message to search for similar issues."
        ),
        FunctionTool.from_defaults(
            fn=explorer.issue_ops.get_issue_closing_info, 
            name="get_issue_closing_info",
            description="Get detailed information about who closed an issue and with what commit/PR"
        ),
        FunctionTool.from_defaults(
            fn=explorer.issue_ops.get_open_issues_related_to_commit, 
            name="get_open_issues_related_to_commit",
            description="Find open issues that might be related to changes in a specific commit"
        ),
        FunctionTool.from_defaults(
            fn=explorer.issue_ops.find_issues_related_to_file, 
            name="find_issues_related_to_file",
            description="Finds issues whose resolution involved changes to the specified file path."
        ),
        FunctionTool.from_defaults(
            fn=explorer.issue_ops.get_issue_resolution_summary, 
            name="get_issue_resolution_summary",
            description="Summarizes how a specific issue was resolved, including linked PRs and a summary of changes."
        ),
        # ASYNC TOOLS - Use wrapper for proper async handling
        FunctionTool.from_defaults(
            fn=create_async_tool_wrapper(explorer.issue_ops.check_issue_status_and_linked_pr, "check_issue_status_and_linked_pr"), 
            name="check_issue_status_and_linked_pr",
            description="Checks the current status (open/closed) of a GitHub issue and lists any directly linked Pull Requests (both merged and open)."
        ),
        FunctionTool.from_defaults(
            fn=create_async_tool_wrapper(explorer.issue_ops.regression_detector, "regression_detector"), 
            name="regression_detector",
            description="Detect if a new issue is a regression of a past one by analyzing similar closed issues"
        ),

        # PROperations (delegates to IssueAwareRAG, GitHistoryTools, LLM)
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.get_pr_for_issue, 
            name="get_pr_for_issue",
            description="Find the pull request associated with a given issue number"
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.get_pr_diff, 
            name="get_pr_diff",
            description="Retrieve the diff for a given pull request number. Use pr_number parameter with the PR number."
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.get_files_changed_in_pr, 
            name="get_files_changed_in_pr",
            description="Lists all files that were modified, added, or deleted in a given pull request. Use pr_number parameter with the PR number."
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.get_pr_summary, 
            name="get_pr_summary",
            description="Provides a concise summary of the changes made in a specific pull request, based on its diff. Use pr_number parameter with the PR number."
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.find_open_prs_for_issue, 
            name="find_open_prs_for_issue",
            description="Find open pull requests that are related to or reference a specific issue number."
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.get_open_pr_status, 
            name="get_open_pr_status",
            description="Get comprehensive status information for an open PR including reviews, CI status, and mergeability."
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.find_open_prs_by_files, 
            name="find_open_prs_by_files",
            description="Find open pull requests that modify specific files in the repository."
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.search_open_prs, 
            name="search_open_prs",
            description="Search through open pull requests by keywords, features, or descriptions to find relevant ones."
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.check_pr_readiness, 
            name="check_pr_readiness",
            description="Check if an open pull request is ready for merging based on reviews, CI status, and conflicts."
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.find_feature_introducing_pr,
            name="find_feature_introducing_pr",
            description="Find which PR/issue introduced a specific feature by searching issue/PR data"
        ),
        # New comprehensive PR analysis tools
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.get_pr_details_from_github,
            name="get_pr_details_from_github",
            description="Get comprehensive PR details directly from GitHub API including reviews, status checks, and metadata"
        ),
        FunctionTool.from_defaults(
            fn=explorer.pr_ops.get_pr_analysis,
            name="get_pr_analysis",
            description="Get complete PR analysis combining local diff data and GitHub metadata - use this as a one-stop tool for understanding what a PR does"
        ),
    ]
    
    return tools

def create_tools_for_subset(explorer: 'AgenticCodebaseExplorer', subset_name: str) -> List[FunctionTool]:
    """
    Creates a subset of tools based on the query type for performance optimization.
    
    Args:
        explorer: AgenticCodebaseExplorer instance
        subset_name: Name of the tool subset (search, file, git, issue, pr, code_gen, comprehensive)
        
    Returns:
        List of FunctionTool instances for the specified subset
    """
    if subset_name not in TOOL_SUBSETS:
        logger.warning(f"Unknown tool subset '{subset_name}', falling back to comprehensive")
        return create_all_tools(explorer)
    
    if subset_name == "comprehensive":
        return create_all_tools(explorer)
    
    # Get all tools first
    all_tools = create_all_tools(explorer)
    
    # Create a mapping of tool names to tool objects
    tool_map = {tool.metadata.name: tool for tool in all_tools}
    
    # Get the subset tool names
    subset_tool_names = set(TOOL_SUBSETS[subset_name]["tools"])
    
    # Always include essential tools
    subset_tool_names.update(ESSENTIAL_TOOLS)
    
    # Filter tools based on subset
    subset_tools = [tool_map[name] for name in subset_tool_names if name in tool_map]
    
    logger.info(f"Created tool subset '{subset_name}' with {len(subset_tools)} tools: {[t.metadata.name for t in subset_tools]}")
    
    return subset_tools

def get_subset_for_query_type(query_type: str) -> str:
    """
    Determine the appropriate tool subset based on query analysis.
    
    Args:
        query_type: The type of query (from query analysis)
        
    Returns:
        Tool subset name to use
    """
    # Mapping from query types to tool subsets
    query_to_subset = {
        "search": "search",
        "find": "search", 
        "locate": "search",
        "file_content": "file",
        "file_structure": "file",
        "directory": "file",
        "git_history": "git",
        "git_blame": "git",
        "commit": "git",
        "who_changed": "git",
        "when_added": "git",
        "issue_analysis": "issue",
        "bug_report": "issue",
        "feature_request": "issue",
        "pr_analysis": "pr",
        "pull_request": "pr",
        "code_review": "pr",
        "code_generation": "code_gen",
        "write_code": "code_gen",
        "example": "code_gen",
        "complex": "comprehensive",
        "multi_step": "comprehensive",
        "unknown": "comprehensive"
    }
    
    return query_to_subset.get(query_type, "comprehensive")
