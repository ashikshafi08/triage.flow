# src/agent_tools/core.py

import os
import json
import asyncio
import time
from typing import List, Dict, Any, Optional, Annotated, AsyncGenerator, TYPE_CHECKING
from pathlib import Path
import logging
import io
import sys
import contextlib
import re
import subprocess
from functools import lru_cache
from datetime import datetime
import concurrent.futures

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer

# Relative imports for components within the src directory
from ..config import settings
from ..chunk_store import ChunkStoreFactory
from ..git_tools import GitBlameTools, GitHistoryTools, IssueClosingTools
from ..commit_index import CommitIndexManager

# Relative imports for components within the agent_tools package
from .prompts import DEFAULT_SYSTEM_PROMPT, COMMIT_INDEX_SYSTEM_PROMPT
from .llm_config import get_llm_instance
from .utilities import (
    get_current_head_sha,
    extract_repo_info,
    blame_line_cached,
    chunk_large_output,
    get_repo_url_from_path 
)
from .file_operations import FileOperations
from .search_operations import SearchOperations
from .code_generation import CodeGenerationOperations
from .git_operations import GitOperations
from .issue_operations import IssueOperations 
from .pr_operations import PROperations
from .tool_registry import create_all_tools # Import for tool creation
from .response_handling import (
    parse_react_steps,
    format_agentic_response,
    clean_captured_output,
    extract_clean_answer,
    basic_cleanup,
    get_natural_exploration_suggestions,
    get_natural_error_recovery
)

# Import context management components
from .context_manager import ContextManager
from .context_aware_tools import ContextAwareToolFactory
from .query_processor import QueryProcessor
from .tool_registry import create_tools_for_subset, get_subset_for_query_type

if TYPE_CHECKING:
    from ..issue_rag import IssueAwareRAG 
    from .pr_operations import PROperations 

logger = logging.getLogger(__name__)

class AgentPool:
    """
    Agent pool for reusing agent instances to reduce creation overhead.
    Maintains pre-created agents for common query types and complexities.
    """
    
    def __init__(self, explorer_instance):
        self.explorer = explorer_instance
        self._simple_agent = None
        self._moderate_agent = None
        self._complex_agent = None
        self._cached_tools_by_subset = {}
        
        # Track agent usage for optimization
        self._usage_stats = {
            "simple": 0,
            "moderate": 0, 
            "complex": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def get_agent_for_query(self, query: str, complexity: str, max_iterations: int, tool_subset: str = None) -> ReActAgent:
        """
        Get an optimized agent for the given query characteristics.
        Uses cached agents when possible and creates new ones only when necessary.
        """
        # Determine if we can reuse an existing agent
        agent_key = self._get_agent_key(complexity, max_iterations, tool_subset)
        
        # For simple queries with standard iterations, use cached agent
        if complexity == "simple" and max_iterations <= 15 and not tool_subset:
            if not self._simple_agent:
                logger.debug("Creating cached simple agent")
                self._simple_agent = self._create_optimized_agent(
                    max_iterations=15, 
                    tool_subset="search",  # Simple queries usually involve search
                    system_prompt=DEFAULT_SYSTEM_PROMPT
                )
            
            # Update iteration limit if needed
            if self._simple_agent.max_iterations != max_iterations:
                self._simple_agent.max_iterations = max_iterations
                
            self._usage_stats["simple"] += 1
            self._usage_stats["cache_hits"] += 1
            return self._simple_agent
        
        # For moderate complexity queries
        elif complexity in ["moderate"] and max_iterations <= 25 and not tool_subset:
            if not self._moderate_agent:
                logger.debug("Creating cached moderate agent")
                self._moderate_agent = self._create_optimized_agent(
                    max_iterations=25,
                    tool_subset=None,  # Use all tools for moderate complexity
                    system_prompt=DEFAULT_SYSTEM_PROMPT
                )
            
            # Update iteration limit if needed
            if self._moderate_agent.max_iterations != max_iterations:
                self._moderate_agent.max_iterations = max_iterations
                
            self._usage_stats["moderate"] += 1
            self._usage_stats["cache_hits"] += 1
            return self._moderate_agent
        
        # For complex queries or custom configurations, create new agent
        else:
            logger.debug(f"Creating new agent for complex query (complexity: {complexity}, iterations: {max_iterations})")
            self._usage_stats["complex"] += 1
            self._usage_stats["cache_misses"] += 1
            return self._create_optimized_agent(
                max_iterations=max_iterations,
                tool_subset=tool_subset,
                system_prompt=DEFAULT_SYSTEM_PROMPT
            )
    
    def _create_optimized_agent(self, max_iterations: int, tool_subset: str = None, system_prompt: str = None) -> ReActAgent:
        """
        Create an optimized agent with optional tool subset for better performance.
        """
        # Use tool subset if specified, otherwise use all tools
        if tool_subset and tool_subset != "comprehensive":
            if tool_subset not in self._cached_tools_by_subset:
                logger.debug(f"Creating tool subset '{tool_subset}' for optimized agent")
                self._cached_tools_by_subset[tool_subset] = create_tools_for_subset(self.explorer, tool_subset)
            tools = self._cached_tools_by_subset[tool_subset]
        else:
            tools = self.explorer.tools
        
        return ReActAgent.from_tools(
            tools=tools,
            llm=self.explorer.base_llm,
            memory=self.explorer.memory,
            verbose=True,
            max_iterations=max_iterations,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT
        )
    
    def _get_agent_key(self, complexity: str, max_iterations: int, tool_subset: str) -> str:
        """Generate a key for agent caching decisions"""
        return f"{complexity}_{max_iterations}_{tool_subset or 'all'}"
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for optimization insights"""
        total_requests = sum(self._usage_stats[k] for k in ["simple", "moderate", "complex"])
        cache_hit_rate = (self._usage_stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self._usage_stats,
            "total_requests": total_requests,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%"
        }
    
    def clear_cache(self):
        """Clear all cached agents (useful for testing or memory management)"""
        self._simple_agent = None
        self._moderate_agent = None
        self._complex_agent = None
        self._cached_tools_by_subset = {}
        logger.info("Agent pool cache cleared")

@contextlib.contextmanager
def capture_output():
    """Capture stdout and stderr during execution"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        yield stdout_buffer, stderr_buffer
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

class AgenticCodebaseExplorer:
    """
    Agentic system for exploring and analyzing codebases using LlamaIndex tools
    with enhanced context sharing between tools following Cognition AI principles
    """
    
    def __init__(self, session_id: str, repo_path: str, issue_rag_system: Optional['IssueAwareRAG'] = None):
        self.session_id = session_id
        self.repo_path = Path(repo_path)
        self.issue_rag_system = issue_rag_system
        self.chunk_store = ChunkStoreFactory.get_instance()
        
        # Initialize context manager for enhanced tool coordination
        self.context_manager = ContextManager(session_id, self.repo_path)
        
        self.file_ops = FileOperations(self.repo_path, self.chunk_store)
        self.search_ops = SearchOperations(self.repo_path)
        self.code_gen_ops = CodeGenerationOperations(self.repo_path) 
        
        self.git_blame_tools = GitBlameTools(str(self.repo_path))
        self.git_history_tools = GitHistoryTools(str(self.repo_path))
        self.issue_closing_tools = IssueClosingTools(str(self.repo_path), issue_rag_system)
        
        repo_owner, repo_name = extract_repo_info(self.repo_path) if self.repo_path else (None, None)
        self.commit_index_manager = CommitIndexManager(
            str(self.repo_path), 
            repo_owner=repo_owner, 
            repo_name=repo_name
        )

        self.git_ops = GitOperations(
            git_blame_tools=self.git_blame_tools,
            git_history_tools=self.git_history_tools,
            commit_index_manager=self.commit_index_manager,
            get_current_head_sha_func=self._get_current_head_sha, 
            chunk_large_output_func=self._chunk_large_output
        )
        
        # Two–tier LLM setup: cheap reasoning model for iterative ReAct steps,
        # higher-quality model for final synthesis / code generation.
        self.base_llm = get_llm_instance(default_model=settings.cheap_model)
        self.final_llm = get_llm_instance()  # uses settings.default_model

        # Keep legacy attribute name for downstream components
        self.llm = self.base_llm

        # Code generation operations benefit from higher-quality model
        self.code_gen_ops.llm = self.final_llm 

        self.pr_ops = PROperations(
            repo_path=self.repo_path,
            issue_rag_system=self.issue_rag_system,
            git_history_tools=self.git_history_tools,
            llm_instance=self.llm,
            chunk_large_output_func=self._chunk_large_output,
            extract_repo_info_func=extract_repo_info 
        )

        self.issue_ops = IssueOperations(
            repo_path=self.repo_path,
            issue_rag_system=self.issue_rag_system,
            issue_closing_tools=self.issue_closing_tools,
            search_ops=self.search_ops,
            # pr_ops=self.pr_ops, # Pass pr_ops if IssueOperations needs it
            get_repo_url_from_path_func=get_repo_url_from_path 
        )
        # If IssueOperations needs pr_ops, it should be passed in its constructor or set here.
        # For example, if IssueOperations has a setter or an attribute:
        # self.issue_ops.pr_operations_dependency = self.pr_ops

        # Initialize query processor for smart query analysis
        self.query_processor = QueryProcessor()
        self.query_processor.set_llm(self.base_llm)
        
        # Create context-aware tools instead of standard tools
        self.context_aware_factory = ContextAwareToolFactory(self.context_manager)
        
        # Check if context-aware tools are enabled (can be disabled for debugging)
        if getattr(settings, 'ENABLE_CONTEXT_AWARE_TOOLS', True):
            logger.info(f"Creating context-aware tools for session {session_id}")
            self.tools = self.context_aware_factory.create_context_aware_tools(self)
        else:
            logger.info(f"Using standard tools for session {session_id}")
            self.tools = create_all_tools(self)
        
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        
        # Initialize agent pool for performance optimization
        self.agent_pool = AgentPool(self)
        
        # Use cost-efficient model for the reasoning loop.
        # Create agent lazily to avoid function calling validation issues during initialization
        self.agent = None
        self._agent_creation_params = {
            "tools": self.tools,
            "llm": self.base_llm,
            "memory": self.memory,
            "verbose": True,
            "max_iterations": settings.AGENTIC_MAX_ITERATIONS,
            "system_prompt": DEFAULT_SYSTEM_PROMPT
        }

    def _ensure_agent_created(self):
        """Ensure the default agent is created when needed"""
        if self.agent is None:
            try:
                self.agent = ReActAgent.from_tools(**self._agent_creation_params)
                logger.info("Default ReAct agent created successfully")
            except Exception as e:
                logger.error(f"Failed to create default agent: {e}")
                # Fallback: create a minimal agent or use agent pool
                logger.info("Falling back to agent pool for all operations")
                self.agent = None  # Will rely on agent pool

    async def initialize_commit_index(
        self, 
        max_commits: Optional[int] = None,
        force_rebuild: bool = False
    ) -> None:
        try:
            await self.commit_index_manager.initialize(
                max_commits=max_commits,
                force_rebuild=force_rebuild
            )
            logger.info("Commit index initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize commit index: {e}")
        
    def _calculate_dynamic_iterations(self, query: str) -> int:
        """Calculate dynamic iteration limit based on query complexity like professional tools"""
        base_iterations = settings.AGENTIC_MAX_ITERATIONS
        
        if not settings.AGENTIC_DYNAMIC_ITERATIONS:
            return base_iterations
        
        # Analyze query complexity
        query_lower = query.lower()
        word_count = len(query.split())
        
        complexity_score = 0
        
        # Word count factor
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # File mentions (like @folder/examples)
        file_mentions = query.count('@')
        complexity_score += file_mentions * 0.5
        
        # Complex operation indicators
        complex_patterns = [
            'integrate', 'trace', 'show', 'analyze', 'compare', 'find all',
            'commits that', 'file contents', 'relationship', 'dependency',
            'evolution', 'history', 'timeline', 'comprehensive'
        ]
        pattern_matches = sum(1 for pattern in complex_patterns if pattern in query_lower)
        complexity_score += pattern_matches * 0.8
        
        # Multiple questions/requirements
        question_indicators = query.count('?') + query.count(' and ') + query.count(' show ')
        complexity_score += question_indicators * 0.7
        
        # Calculate final iterations
        if complexity_score >= 3:
            # High complexity - use multiplier
            iterations = int(base_iterations * settings.AGENTIC_COMPLEXITY_MULTIPLIER)
        elif complexity_score >= 2:
            # Medium complexity
            iterations = int(base_iterations * 1.5)
        else:
            # Simple query
            iterations = base_iterations
        
        # Cap at reasonable limits
        return min(max(iterations, settings.AGENTIC_MIN_ITERATIONS), 100)
    
    def create_enhanced_agent(self, query: str = "", max_iterations: int = None) -> ReActAgent:
        """Create a specialized agent with dynamic iteration limit for complex analysis"""
        if max_iterations is None:
            max_iterations = self._calculate_dynamic_iterations(query) if query else settings.AGENTIC_MAX_ITERATIONS
        
        logger.info(f"Creating enhanced agent with {max_iterations} iterations for query complexity")
        
        return ReActAgent.from_tools(
            tools=self.tools,
            llm=self.base_llm,
            memory=self.memory,
            verbose=True,
            max_iterations=max_iterations,
            system_prompt=DEFAULT_SYSTEM_PROMPT
        )
        
    async def query(self, user_message: str, use_enhanced_agent: bool = False) -> str:
        try:
            logger.info(f"Starting optimized agentic analysis: {user_message[:100]}...")
            
            # Start execution context for this query
            execution_context = self.context_manager.start_execution(user_message)
            
            # Use smart query analysis for optimization
            query_info = self.query_processor.analyze_query(user_message)
            complexity = query_info.complexity
            query_type = query_info.query_type
            max_iterations = query_info.max_iterations
            
            # Determine optimal tool subset based on query type
            tool_subset = get_subset_for_query_type(query_type) if complexity in ["simple", "moderate"] else None
            
            logger.info(f"Query analysis: complexity={complexity}, type={query_type}, iterations={max_iterations}, tools={tool_subset}")
            
            with capture_output() as (stdout_buffer, stderr_buffer):
                if use_enhanced_agent or complexity in ["complex", "research"]:
                    # Use optimized agent from pool for complex queries
                    optimized_agent = self.agent_pool.get_agent_for_query(
                        user_message, complexity, max_iterations, tool_subset
                    )
                    response = await optimized_agent.achat(user_message)
                else:
                    # Use agent pool for simple/moderate queries too
                    optimized_agent = self.agent_pool.get_agent_for_query(
                        user_message, complexity, max_iterations, tool_subset
                    )
                    response = await optimized_agent.achat(user_message)
            
            logger.info(f"Optimized agentic analysis completed successfully")
            
            # Log performance statistics
            pool_stats = self.agent_pool.get_usage_stats()
            logger.info(f"Agent pool stats: {pool_stats}")
            
            # Get execution summary for logging
            summary = self.context_manager.get_execution_summary()
            logger.info(f"Execution summary: {summary}")
            
            captured_output = stdout_buffer.getvalue() or stderr_buffer.getvalue()
            captured_output = clean_captured_output(captured_output)
            full_react_trace = captured_output
            if not ("Thought:" in captured_output or "Action:" in captured_output):
                # Ensure the agent is created before accessing memory
                self._ensure_agent_created()
                if self.agent:
                    chat_history = self.agent.memory.get_all()
                    for msg in reversed(chat_history):
                        if hasattr(msg, 'role') and msg.role.value == "assistant":
                            full_react_trace = msg.content
                            break
                if not full_react_trace: full_react_trace = str(response)
            steps, final_answer = parse_react_steps(full_react_trace)
            if not final_answer and steps and steps[-1]['type'] == 'observation':
                final_answer = str(steps[-1]['content'])
            if not final_answer and not steps:
                 final_answer = str(response)
            suggestions = [] 
            
            # Enhance final answer with context information if available
            enhanced_final_answer = self._enhance_final_answer_with_context(final_answer, execution_context)
            
            return format_agentic_response(
                steps, 
                enhanced_final_answer, 
                partial=False, 
                suggestions=suggestions,
                repo_path=str(self.repo_path) if hasattr(self, 'repo_path') else None,
                user_query=user_message
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in agentic query: {error_msg}")
            return format_agentic_response([], final_answer=f"Error: {error_msg}", partial=True)

    async def stream_query(self, user_message: str):
        try:
            logger.info(f"[stream] Starting agentic analysis: {user_message[:100]}...")
            
            # Start execution context for this query
            execution_context = self.context_manager.start_execution(user_message)
            
            # Calculate dynamic iterations for professional-grade handling
            dynamic_iterations = self._calculate_dynamic_iterations(user_message)
            logger.info(f"[stream] Using {dynamic_iterations} iterations for query complexity")
            
            yield json.dumps({
                "type": "status", 
                "content": f"Starting analysis with {dynamic_iterations} max iterations...", 
                "step": 0
            })
            
            with capture_output() as (stdout_buffer, stderr_buffer):
                # Use appropriate agent based on complexity
                if dynamic_iterations > settings.AGENTIC_MAX_ITERATIONS:
                    # Complex query - use enhanced agent
                    enhanced_agent = self.create_enhanced_agent(user_message)
                    response = await enhanced_agent.achat(user_message)
                else:
                    # Simple to medium complexity - use optimized agent
                    if dynamic_iterations != settings.AGENTIC_MAX_ITERATIONS:
                        # Create agent with adjusted iterations
                        temp_agent = ReActAgent.from_tools(
                            tools=self.tools,
                            llm=self.base_llm,
                            memory=self.memory,
                            verbose=True,
                            max_iterations=dynamic_iterations,
                            system_prompt=DEFAULT_SYSTEM_PROMPT
                        )
                        response = await temp_agent.achat(user_message)
                    else:
                        # Ensure the default agent is created
                        self._ensure_agent_created()
                        response = await self.agent.achat(user_message)
            
            # Get execution summary
            summary = self.context_manager.get_execution_summary()
            logger.info(f"[stream] Execution summary: {summary}")
            
            captured_output = stdout_buffer.getvalue() or stderr_buffer.getvalue()
            captured_output = clean_captured_output(captured_output)
            full_react_trace = captured_output
            if not ("Thought:" in captured_output or "Action:" in captured_output):
                # Ensure the agent is created before accessing memory
                self._ensure_agent_created()
                if self.agent:
                    chat_history = self.agent.memory.get_all()
                    for msg in reversed(chat_history):
                        if hasattr(msg, 'role') and msg.role.value == "assistant":
                            full_react_trace = msg.content
                            break
                if not full_react_trace: full_react_trace = str(response)
            steps, final_answer = parse_react_steps(full_react_trace)
            if not final_answer and steps and steps[-1]['type'] == 'observation':
                final_answer = str(steps[-1]['content'])
            if not final_answer and not steps:
                 final_answer = str(response)
            for i, step in enumerate(steps):
                yield json.dumps({"type": "step", "step": step})
                await asyncio.sleep(0.01)
            suggestions = [] 
            
            # Enhance final answer with context information
            enhanced_final_answer = self._enhance_final_answer_with_context(final_answer, execution_context)
            
            # Apply professional formatting for streaming response
            formatted_response = format_agentic_response(
                steps, 
                enhanced_final_answer, 
                partial=False, 
                suggestions=suggestions or [],
                repo_path=str(self.repo_path) if hasattr(self, 'repo_path') else None,
                user_query=user_message
            )
            
            # Parse the formatted response and add stream-specific fields
            formatted_data = json.loads(formatted_response)
            final_payload = {
                **formatted_data,
                "final": True,
                "total_steps": len(steps),
                "execution_summary": summary  # Include execution summary in stream
            }
            yield json.dumps(final_payload)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[stream] Error in agentic stream_query: {error_msg}")
            yield json.dumps({
                "type": "error", "final": True, "steps": [], "final_answer": None,
                "partial": True, "suggestions": ["Try a more specific question"], "error": error_msg
            })

    def reset_memory(self):
        """Reset both agent memory and context manager"""
        self.memory.reset()
        if hasattr(self, 'context_manager'):
            self.context_manager.cleanup()

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution context"""
        if hasattr(self, 'context_manager'):
            return self.context_manager.get_execution_summary()
        return {"status": "context_manager_not_available"}

    def _enhance_final_answer_with_context(self, final_answer: str, execution_context) -> str:
        """Enhance the final answer with context information if beneficial"""
        try:
            if not execution_context or not hasattr(self, 'context_manager'):
                return final_answer
            
            summary = self.context_manager.get_execution_summary()
            
            # Only add context summary if there was significant tool usage
            if summary.get("total_executions", 0) > 3:
                context_info = f"\n\n---\n**Analysis Context:**\n"
                context_info += f"- Tools used: {', '.join(summary.get('tools_used', []))}\n"
                context_info += f"- Files analyzed: {summary.get('files_discovered', 0)}\n"
                context_info += f"- Execution steps: {summary.get('total_executions', 0)}\n"
                
                if summary.get("conflicts_resolved", 0) > 0:
                    context_info += f"- Conflicts resolved: {summary.get('conflicts_resolved', 0)}\n"
                
                return final_answer + context_info
            
            return final_answer
            
        except Exception as e:
            logger.warning(f"Error enhancing final answer with context: {e}")
            return final_answer

    def _get_current_head_sha(self) -> Optional[str]: 
        return get_current_head_sha(self.repo_path)

    def _chunk_large_output(self, content: str) -> str: 
        return chunk_large_output(content, self.chunk_store)

    def find_issue_related_files(self, issue_description: str, search_depth: str = "surface") -> str:
        """Delegate to issue operations for finding files related to issues."""
        return self.issue_ops.find_issue_related_files(issue_description, search_depth)

    def analyze_file_structure(self, file_path: str) -> str:
        """Delegate to file operations for analyzing file structure."""
        return self.file_ops.analyze_file_structure(file_path)

    def read_file(self, file_path: str) -> str:
        """Delegate to file operations for reading a file's content."""
        return self.file_ops.read_file(file_path)

    def _extract_technical_requirements(self, issue_data):
        """Extract technical requirements from issue data for enhanced analysis."""
        pass

    def _extract_issue_keywords(self, text: str):
        """Extract keywords from issue text for targeted search."""
        pass
