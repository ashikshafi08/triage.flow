# src/agent_tools/pr_operations.py

import json
import logging
import asyncio
import re
import time
import subprocess
from typing import List, Dict, Any, Optional, Annotated, TYPE_CHECKING
from pathlib import Path
import concurrent.futures

if TYPE_CHECKING:
    from ..issue_rag import IssueAwareRAG
    from ..git_tools import GitHistoryTools
    from llama_index.core.llms import LLM
    from ..config import Settings # For typing, if settings are passed
    from .utilities import chunk_large_output_func, extract_repo_info_func # Renaming

# For standalone utilities if needed
from .utilities import chunk_large_output as chunk_large_output_util
from .utilities import extract_repo_info as extract_repo_info_util

# Assuming settings might be needed for summarization model
try:
    from ..config import settings
    from llama_index.llms.openrouter import OpenRouter # For get_pr_summary
except ImportError:
    class MockSettings:
        openrouter_api_key = None
        summarization_model = None
    settings = MockSettings()
    # Mock OpenRouter if llama_index is not fully available in a test/isolated context
    class OpenRouter:
        def __init__(self, *args, **kwargs): pass
        def complete(self, prompt: str):
            class MockResponse: text = f"Mock summary for: {prompt[:50]}"
            return MockResponse()


logger = logging.getLogger(__name__)

class PROperations:
    def __init__(self, 
                 repo_path: Path,
                 issue_rag_system: Optional['IssueAwareRAG'],
                 git_history_tools: 'GitHistoryTools',
                 llm_instance: 'LLM',
                 # Pass utility functions
                 chunk_large_output_func: callable,
                 extract_repo_info_func: callable
                ):
        self.repo_path = repo_path
        self.issue_rag_system = issue_rag_system
        self.git_history_tools = git_history_tools
        self.llm = llm_instance
        self._chunk_large_output = chunk_large_output_func
        self._extract_repo_info = extract_repo_info_func


    def get_pr_for_issue(self, issue_number: Annotated[int, "Issue number"]) -> str:
        if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'patch_builder'):
            return json.dumps({"error": "Issue RAG or PatchLinkageBuilder not available."})
        patch_builder = self.issue_rag_system.indexer.patch_builder
        links = patch_builder.load_patch_links().get(issue_number, [])
        return json.dumps({"issue_number": issue_number, "found_prs": [l.to_dict() for l in links]})

    def get_pr_diff(self, pr_number: Annotated[int, "PR number"]) -> str:
        if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'diff_docs'):
            return json.dumps({"error": "Issue RAG or diff_docs not available."})
        diff_doc = self.issue_rag_system.indexer.diff_docs.get(pr_number)
        if not diff_doc: return json.dumps({"error": f"No cached diff for PR #{pr_number}."})
        
        diff_path = Path(diff_doc.diff_path) # diff_doc.diff_path should be absolute or resolvable
        if not diff_path.exists():
             return json.dumps({"error": f"Diff file not found at {diff_doc.diff_path}."})
        try:
            diff_text = diff_path.read_text(encoding='utf-8', errors='ignore')
            return json.dumps({"pr_number": pr_number, "diff_summary": diff_doc.diff_summary, "full_diff": diff_text})
        except Exception as e: return json.dumps({"error": f"Error reading diff: {e}"})

    def get_files_changed_in_pr(self, pr_number: Annotated[int, "PR number"]) -> str:
        if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'diff_docs'):
            return json.dumps({"error": "Issue RAG or diff_docs not available."})
        diff_doc = self.issue_rag_system.indexer.diff_docs.get(pr_number)
        if not diff_doc: return json.dumps({"error": f"No cached diff for PR #{pr_number}."})
        return json.dumps({"pr_number": pr_number, "files_changed": diff_doc.files_changed})

    def get_pr_summary(self, pr_number: Annotated[int, "PR number"]) -> str:
        # Simplified: original logic for LLM summarization needs careful porting
        # For now, returns pre-extracted summary if available.
        if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'diff_docs'):
            return json.dumps({"error": "Issue RAG or diff_docs not available."})
        diff_doc = self.issue_rag_system.indexer.diff_docs.get(pr_number)
        if not diff_doc: return json.dumps({"error": f"No cached diff for PR #{pr_number}."})
        
        # Placeholder for full LLM summarization logic from original file
        # This would involve reading the diff_text and using self.llm or a dedicated summarization LLM
        # For now, just returning the pre-extracted summary.
        return json.dumps({
            "pr_number": pr_number, 
            "summary": diff_doc.diff_summary or "Summary not available (placeholder).",
            "files_changed": diff_doc.files_changed
        })

    async def find_open_prs_for_issue(self, issue_number: Annotated[int, "Issue number"]) -> str:
        # Simplified placeholder. Original logic involved PatchLinkageBuilder and text similarity.
        logger.warning("find_open_prs_for_issue is a placeholder in PROperations.")
        if not self.issue_rag_system or not self.issue_rag_system.is_initialized():
            return json.dumps({"error": "Issue RAG system not initialized", "open_prs": []})
        # Conceptual: actual implementation would search open PRs from indexer
        return json.dumps({"message": f"Placeholder search for open PRs for issue #{issue_number}", "open_prs": []})

    def get_open_pr_status(self, pr_number: Annotated[int, "PR number"]) -> str:
        logger.warning("get_open_pr_status is a placeholder in PROperations.")
        if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'open_pr_docs'):
            return json.dumps({"error": "Issue RAG or open_pr_docs not available."})
        pr_doc = self.issue_rag_system.indexer.open_pr_docs.get(pr_number)
        if not pr_doc: return json.dumps({"error": f"Open PR #{pr_number} not found in index."})
        # Assuming pr_doc has attributes like title, author, review_decision, etc.
        return json.dumps(pr_doc.to_dict() if hasattr(pr_doc, 'to_dict') else vars(pr_doc), indent=2)


    def find_open_prs_by_files(self, file_paths: Annotated[List[str], "List of file paths"]) -> str:
        logger.warning("find_open_prs_by_files is a placeholder in PROperations.")
        # Conceptual: actual implementation would search open PRs from indexer
        return json.dumps({"message": "Placeholder for find_open_prs_by_files", "files": file_paths, "found_prs": []})

    async def search_open_prs(self, query: Annotated[str, "Search query"], limit: Annotated[int, "Limit"] = 5) -> str:
        logger.warning("search_open_prs is a placeholder in PROperations.")
        # Conceptual: actual implementation would search open PRs from indexer
        return json.dumps({"message": "Placeholder for search_open_prs", "query": query, "found_prs": []})

    def check_pr_readiness(self, pr_number: Annotated[int, "PR number"]) -> str:
        logger.warning("check_pr_readiness is a placeholder in PROperations.")
        # Conceptual: actual implementation would use get_open_pr_status and evaluate
        return json.dumps({"message": "Placeholder for check_pr_readiness", "pr_number": pr_number, "status": "Unknown"})

    def find_feature_introducing_pr(self, feature_name: Annotated[str, "Feature name"]) -> str:
        # This is a complex method. For now, a simplified placeholder.
        # Original logic involved RAG, git commit search, and diff content search.
        logger.warning("find_feature_introducing_pr is a placeholder in PROperations.")
        # Try a very basic git log search as a placeholder
        try:
            git_cmd = ["git", "log", "--grep", feature_name, "--pretty=format:%H|%s", "-10", "--all"]
            result = subprocess.run(git_cmd, capture_output=True, text=True, cwd=self.repo_path)
            prs = []
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    parts = line.split('|',1)
                    commit_sha, subject = parts[0], parts[1]
                    pr_match = re.search(r'#(\d+)', subject)
                    if pr_match:
                        prs.append({"pr_number": int(pr_match.group(1)), "title": subject, "commit_sha": commit_sha, "source": "git_log_grep"})
            if prs:
                return json.dumps({"feature_name": feature_name, "most_likely_introducing_pr": prs[0], "all_related_prs": prs[:3]}, indent=2)
            return json.dumps({"message": f"No direct PR found for '{feature_name}' via simple grep.", "feature_name": feature_name})
        except Exception as e:
            return json.dumps({"error": str(e)})
