"""Agentic codebase explorer - tinygrad-style rewrite (963→400 lines)"""
import os, json, asyncio, time, logging, io, sys, contextlib, re
from typing import List, Dict, Any, Optional, AsyncGenerator, TYPE_CHECKING
from pathlib import Path
from functools import lru_cache
from datetime import datetime

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

from ..config import settings
from ..chunk_store import ChunkStoreFactory
from ..git_tools import GitBlameTools, GitHistoryTools, IssueClosingTools
from ..commit_index import CommitIndexManager

from .prompts import DEFAULT_SYSTEM_PROMPT
from .llm_config import get_llm_instance
from .utilities import get_current_head_sha, extract_repo_info, blame_line_cached, chunk_large_output, get_repo_url_from_path
from .file_operations import FileOperations
from .search_operations import SearchOperations
from .code_generation import CodeGenerationOperations
from .git_operations import GitOperations
from .issue_operations import IssueOperations
from .pr_operations import PROperations
from .tool_registry import create_all_tools, create_tools_for_subset, get_subset_for_query_type
from .response_handling import parse_react_steps, format_agentic_response, clean_captured_output
from .context_manager import ContextManager
from .context_aware_tools import ContextAwareToolFactory
from .query_processor import QueryProcessor
from .agent_pool import AgentPool

if TYPE_CHECKING:
    from ..unified_rag import IssueAwareRAG

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def capture_output():
    """Capture stdout/stderr during execution."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    try:
        sys.stdout, sys.stderr = stdout_buf, stderr_buf
        yield stdout_buf, stderr_buf
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


class AgenticCodebaseExplorer:
    """Agentic system for exploring codebases using LlamaIndex tools with learning."""

    def __init__(self, session_id: str, repo_path: str, issue_rag_system: Optional['IssueAwareRAG'] = None):
        self.session_id, self.repo_path = session_id, Path(repo_path)
        self.issue_rag_system = issue_rag_system
        self.chunk_store = ChunkStoreFactory.get_instance()

        # Context and operations
        self.context_manager = ContextManager(session_id, self.repo_path)
        self.file_ops = FileOperations(self.repo_path, self.chunk_store)
        self.search_ops = SearchOperations(self.repo_path)
        self.code_gen_ops = CodeGenerationOperations(self.repo_path)

        # Git tools
        self.git_blame_tools = GitBlameTools(str(self.repo_path))
        self.git_history_tools = GitHistoryTools(str(self.repo_path))
        self.issue_closing_tools = IssueClosingTools(str(self.repo_path), issue_rag_system)

        # Commit index
        owner, name = extract_repo_info(self.repo_path) if self.repo_path else (None, None)
        self.commit_index_manager = CommitIndexManager(str(self.repo_path), repo_owner=owner, repo_name=name)

        self.git_ops = GitOperations(
            git_blame_tools=self.git_blame_tools, git_history_tools=self.git_history_tools,
            commit_index_manager=self.commit_index_manager,
            get_current_head_sha_func=lambda: get_current_head_sha(str(self.repo_path)),
            chunk_large_output_func=lambda o, m: chunk_large_output(o, m)
        )

        # Two-tier LLM: cheap for reasoning, quality for synthesis
        self.base_llm = get_llm_instance(default_model=settings.cheap_model)
        self.final_llm = get_llm_instance()
        self.llm = self.base_llm
        self.code_gen_ops.llm = self.final_llm

        self.pr_ops = PROperations(
            repo_path=self.repo_path, issue_rag_system=self.issue_rag_system,
            git_history_tools=self.git_history_tools, llm_instance=self.llm,
            chunk_large_output_func=lambda o, m: chunk_large_output(o, m),
            extract_repo_info_func=extract_repo_info
        )

        self.issue_ops = IssueOperations(
            repo_path=self.repo_path, issue_rag_system=self.issue_rag_system,
            issue_closing_tools=self.issue_closing_tools, search_ops=self.search_ops,
            get_repo_url_from_path_func=get_repo_url_from_path
        )

        # Query processor and tools
        self.query_processor = QueryProcessor()
        self.query_processor.set_llm(self.base_llm)
        self.context_aware_factory = ContextAwareToolFactory(self.context_manager)

        self.tools = (self.context_aware_factory.create_context_aware_tools(self)
                      if getattr(settings, 'ENABLE_CONTEXT_AWARE_TOOLS', True)
                      else create_all_tools(self))

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        self.agent_pool = AgentPool(self)
        self.agent = None
        self._agent_params = {"tools": self.tools, "llm": self.base_llm, "memory": self.memory,
                              "verbose": True, "max_iterations": settings.AGENTIC_MAX_ITERATIONS,
                              "system_prompt": DEFAULT_SYSTEM_PROMPT}

    def _ensure_agent(self):
        if self.agent is None:
            try: self.agent = ReActAgent.from_tools(**self._agent_params)
            except Exception as e: logger.error(f"Failed to create agent: {e}")

    async def initialize_commit_index(self, max_commits: Optional[int] = None, force_rebuild: bool = False):
        try: await self.commit_index_manager.initialize(max_commits=max_commits, force_rebuild=force_rebuild)
        except Exception as e: logger.warning(f"Failed to initialize commit index: {e}")

    def _calculate_dynamic_iterations(self, query: str) -> int:
        """Calculate iteration limit based on query complexity."""
        base = settings.AGENTIC_MAX_ITERATIONS
        q = query.lower()
        if any(k in q for k in ["analyze", "explain", "compare", "investigate", "how does"]): return min(base + 15, 50)
        if any(k in q for k in ["find", "where", "which", "list"]): return max(base - 5, 10)
        if any(k in q for k in ["review", "audit", "debug", "fix"]): return min(base + 20, 60)
        return base

    def _get_current_head_sha(self) -> Optional[str]:
        return get_current_head_sha(str(self.repo_path))

    def _chunk_large_output(self, output: str, max_chars: int = 8000) -> str:
        return chunk_large_output(output, max_chars)

    def create_enhanced_agent(self, query: str = None, max_iterations: int = None) -> ReActAgent:
        if max_iterations is None:
            max_iterations = self._calculate_dynamic_iterations(query) if query else settings.AGENTIC_MAX_ITERATIONS
        return ReActAgent.from_tools(tools=self.tools, llm=self.base_llm, memory=self.memory,
                                      verbose=True, max_iterations=max_iterations, system_prompt=DEFAULT_SYSTEM_PROMPT)

    async def query(self, user_message: str, use_enhanced_agent: bool = False, enable_learning: bool = True) -> str:
        """Execute query."""
        try:
            return await self._execute_query(user_message, stream=False)
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return format_agentic_response([], final_answer=f"Error: {e}", partial=True)

    async def stream_query(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream query results."""
        try:
            iterations = self._calculate_dynamic_iterations(user_message)
            yield json.dumps({"type": "status", "content": f"Starting analysis with {iterations} iterations...", "step": 0})
            result = await self._execute_query(user_message, stream=True, iterations=iterations)
            yield result
        except Exception as e:
            logger.error(f"Error in stream_query: {e}")
            yield format_agentic_response([], final_answer=f"Error: {e}", partial=True)

    async def _execute_query(self, user_message: str, stream: bool = False, iterations: int = None) -> str:
        """Core query execution logic."""
        execution_context = self.context_manager.start_execution(user_message)
        query_info = self.query_processor.analyze_query(user_message)
        complexity, query_type = query_info.complexity, query_info.query_type
        max_iter = iterations or query_info.max_iterations
        tool_subset = get_subset_for_query_type(query_type) if complexity in ["simple", "moderate"] else None

        logger.info(f"Query: complexity={complexity}, type={query_type}, iterations={max_iter}")

        with capture_output() as (stdout_buf, stderr_buf):
            agent = self.agent_pool.get_agent_for_query(user_message, complexity, max_iter, tool_subset)
            response = await agent.achat(user_message)

        # Parse response
        captured = clean_captured_output(stdout_buf.getvalue() or stderr_buf.getvalue())
        full_trace = captured
        if "Thought:" not in captured and "Action:" not in captured:
            self._ensure_agent()
            if self.agent and (history := self.agent.memory.get_all()):
                for msg in reversed(history):
                    if hasattr(msg, 'role') and msg.role.value == "assistant":
                        full_trace = msg.content; break
        if not full_trace: full_trace = str(response)

        steps, final_answer = parse_react_steps(full_trace)
        if not final_answer and steps and steps[-1]['type'] == 'observation':
            final_answer = str(steps[-1]['content'])
        if not final_answer: final_answer = str(response)

        enhanced = self._enhance_final_answer_with_context(final_answer, execution_context)
        return format_agentic_response(steps, enhanced, partial=False, suggestions=[],
                                        repo_path=str(self.repo_path), user_query=user_message)

    def _enhance_final_answer_with_context(self, answer: str, context) -> str:
        """Enhance answer with execution context."""
        if not context: return answer
        summary = self.context_manager.get_execution_summary()
        if not summary.get('tool_executions'): return answer
        files = [e.get('primary_file') for e in summary['tool_executions'] if e.get('primary_file')]
        if files and "file" not in answer.lower():
            return f"{answer}\n\n**Files examined:** {', '.join(set(files)[:5])}"
        return answer

