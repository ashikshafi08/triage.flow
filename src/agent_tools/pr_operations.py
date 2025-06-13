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
    from ..cache.redis_cache_manager import RedisCacheManager
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
    RedisCacheManager = None


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
        
        # Initialize PR-specific cache
        if RedisCacheManager:
            self.pr_cache = RedisCacheManager(
                namespace="pr_ops",
                default_ttl=1800,  # 30 minutes for PR data
                max_memory_items=500
            )
        else:
            self.pr_cache = None

    async def _get_cached_or_fetch(self, cache_key: str, fetch_func, *args, **kwargs):
        """Get data from cache or fetch and cache it"""
        if not self.pr_cache:
            return await fetch_func(*args, **kwargs) if asyncio.iscoroutinefunction(fetch_func) else fetch_func(*args, **kwargs)
        
        # Try cache first
        cached_result = await self.pr_cache.get(cache_key)
        if cached_result:
            logger.info(f"PR cache hit for key: {cache_key}")
            return cached_result
        
        # Fetch and cache
        logger.info(f"PR cache miss for key: {cache_key}")
        if asyncio.iscoroutinefunction(fetch_func):
            result = await fetch_func(*args, **kwargs)
        else:
            result = fetch_func(*args, **kwargs)
        
        await self.pr_cache.set(cache_key, result)
        return result

    def get_pr_for_issue(self, issue_number: Annotated[int, "Issue number"]) -> str:
        cache_key = f"pr_for_issue:{issue_number}"
        
        def _fetch_pr_for_issue():
            # First try the issue RAG system if available
            if self.issue_rag_system and hasattr(self.issue_rag_system, 'indexer') and hasattr(self.issue_rag_system.indexer, 'patch_builder'):
                try:
                    patch_builder = self.issue_rag_system.indexer.patch_builder
                    links = patch_builder.load_patch_links().get(issue_number, [])
                    if links:
                        return json.dumps({"issue_number": issue_number, "found_prs": [l.to_dict() for l in links]})
                except Exception as e:
                    logger.warning(f"Issue RAG patch_builder failed: {e}")
            
            # Fallback: Use git log to search for PR references
            try:
                # Search commit messages for references to this issue
                git_cmd = [
                    "git", "log", "--grep", f"#{issue_number}", 
                    "--pretty=format:%H|%s", "--all", "-100"
                ]
                result = subprocess.run(git_cmd, capture_output=True, text=True, cwd=self.repo_path)
                
                found_prs = []
                if result.returncode == 0 and result.stdout.strip():
                    for line in result.stdout.strip().split('\n'):
                        parts = line.split('|', 1)
                        if len(parts) >= 2:
                            commit_sha, subject = parts
                            # Look for PR numbers in commit message
                            pr_matches = re.findall(r'(?:Merge pull request #|#)(\d+)', subject)
                            for pr_num in pr_matches:
                                if pr_num not in [str(p['pr_number']) for p in found_prs]:
                                    found_prs.append({
                                        "pr_number": int(pr_num),
                                        "commit_sha": commit_sha,
                                        "commit_subject": subject,
                                        "source": "git_log"
                                    })
                
                if found_prs:
                    return json.dumps({
                        "issue_number": issue_number, 
                        "found_prs": found_prs,
                        "message": f"Found {len(found_prs)} related PR(s) via git log"
                    })
                else:
                    return json.dumps({
                        "issue_number": issue_number,
                        "found_prs": [],
                        "message": "No PRs found for this issue via git log search"
                    })
                    
            except Exception as e:
                logger.error(f"Git log search failed: {e}")
                return json.dumps({"error": f"Failed to search for PRs: {str(e)}"})
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_pr_for_issue))
        else:
            return _fetch_pr_for_issue()

    def get_pr_diff(self, pr_number: Annotated[int, "PR number"]) -> str:
        cache_key = f"pr_diff:{pr_number}"
        
        def _fetch_pr_diff():
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
            except Exception as e: 
                return json.dumps({"error": f"Error reading diff: {e}"})
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_pr_diff))
        else:
            return _fetch_pr_diff()

    def get_files_changed_in_pr(self, pr_number: Annotated[int, "PR number"]) -> str:
        cache_key = f"pr_files:{pr_number}"
        
        def _fetch_pr_files():
            if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'diff_docs'):
                return json.dumps({"error": "Issue RAG or diff_docs not available."})
            diff_doc = self.issue_rag_system.indexer.diff_docs.get(pr_number)
            if not diff_doc: return json.dumps({"error": f"No cached diff for PR #{pr_number}."})
            return json.dumps({"pr_number": pr_number, "files_changed": diff_doc.files_changed})
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_pr_files))
        else:
            return _fetch_pr_files()

    def get_pr_summary(self, pr_number: Annotated[int, "PR number"]) -> str:
        cache_key = f"pr_summary:{pr_number}"
        
        def _fetch_pr_summary():
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
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_pr_summary))
        else:
            return _fetch_pr_summary()

    async def find_open_prs_for_issue(self, issue_number: Annotated[int, "Issue number"]) -> str:
        cache_key = f"open_prs_for_issue:{issue_number}"
        
        async def _fetch_open_prs():
            # Simplified placeholder. Original logic involved PatchLinkageBuilder and text similarity.
            logger.warning("find_open_prs_for_issue is a placeholder in PROperations.")
            if not self.issue_rag_system or not self.issue_rag_system.is_initialized():
                return json.dumps({"error": "Issue RAG system not initialized", "open_prs": []})
            # Conceptual: actual implementation would search open PRs from indexer
            return json.dumps({"message": f"Placeholder search for open PRs for issue #{issue_number}", "open_prs": []})
        
        return await self._get_cached_or_fetch(cache_key, _fetch_open_prs)

    def get_open_pr_status(self, pr_number: Annotated[int, "PR number"]) -> str:
        cache_key = f"open_pr_status:{pr_number}"
        
        def _fetch_pr_status():
            logger.warning("get_open_pr_status is a placeholder in PROperations.")
            if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'open_pr_docs'):
                return json.dumps({"error": "Issue RAG or open_pr_docs not available."})
            pr_doc = self.issue_rag_system.indexer.open_pr_docs.get(pr_number)
            if not pr_doc: return json.dumps({"error": f"Open PR #{pr_number} not found in index."})
            # Assuming pr_doc has attributes like title, author, review_decision, etc.
            return json.dumps(pr_doc.to_dict() if hasattr(pr_doc, 'to_dict') else vars(pr_doc), indent=2)
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_pr_status))
        else:
            return _fetch_pr_status()

    def find_open_prs_by_files(self, file_paths: Annotated[List[str], "List of file paths"]) -> str:
        cache_key = f"prs_by_files:{':'.join(sorted(file_paths))}"
        
        def _fetch_prs_by_files():
            logger.warning("find_open_prs_by_files is a placeholder in PROperations.")
            # Conceptual: actual implementation would search open PRs from indexer
            return json.dumps({"message": "Placeholder for find_open_prs_by_files", "files": file_paths, "found_prs": []})
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_prs_by_files))
        else:
            return _fetch_prs_by_files()

    async def search_open_prs(self, query: Annotated[str, "Search query"], limit: Annotated[int, "Limit"] = 5) -> str:
        cache_key = f"search_prs:{query}:{limit}"
        
        async def _search_prs():
            logger.warning("search_open_prs is a placeholder in PROperations.")
            # Conceptual: actual implementation would search open PRs from indexer
            return json.dumps({"message": "Placeholder for search_open_prs", "query": query, "found_prs": []})
        
        return await self._get_cached_or_fetch(cache_key, _search_prs)

    def check_pr_readiness(self, pr_number: Annotated[int, "PR number"]) -> str:
        cache_key = f"pr_readiness:{pr_number}"
        
        def _check_readiness():
            logger.warning("check_pr_readiness is a placeholder in PROperations.")
            # Conceptual: actual implementation would use get_open_pr_status and evaluate
            return json.dumps({"message": "Placeholder for check_pr_readiness", "pr_number": pr_number, "status": "Unknown"})
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _check_readiness))
        else:
            return _check_readiness()

    def find_feature_introducing_pr(self, feature_name: Annotated[str, "Feature name"]) -> str:
        cache_key = f"feature_pr:{feature_name}"
        
        def _find_feature_pr():
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
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _find_feature_pr))
        else:
            return _find_feature_pr()

    async def get_pr_details_from_github(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Get comprehensive PR details from GitHub API with caching"""
        cache_key = f"github_pr_details:{pr_number}"
        
        async def _fetch_github_pr():
            try:
                from ..github_client import GitHubIssueClient
                from .utilities import get_repo_url_from_path
                
                github_client = GitHubIssueClient()
                repo_url = get_repo_url_from_path(self.repo_path)
                
                if not repo_url:
                    return json.dumps({"error": "Cannot determine repository URL"})
                
                # Get detailed PR info from GitHub
                pr_info = await github_client.get_pr_detailed_info(repo_url, pr_number)
                
                if not pr_info:
                    # Try to provide helpful context about the PR range
                    return json.dumps({
                        "error": f"PR #{pr_number} not found",
                        "suggestion": "This PR number may not exist or may be from a different repository",
                        "repo_url": repo_url
                    })
                
                # Convert to dict for JSON serialization
                pr_dict = {
                    "number": pr_info.number,
                    "title": pr_info.title,
                    "state": pr_info.state,
                    "url": pr_info.url,
                    "body": pr_info.body,
                    "author": pr_info.user.login if pr_info.user else None,
                    "created_at": pr_info.created_at,
                    "updated_at": pr_info.updated_at,
                    "merged_at": pr_info.merged_at,
                    "files_changed": pr_info.files_changed,
                    "review_decision": pr_info.review_decision,
                    "mergeable": pr_info.mergeable,
                    "draft": pr_info.draft,
                    "additions": pr_info.additions,
                    "deletions": pr_info.deletions,
                    "reviews": [
                        {
                            "author": review.author,
                            "state": review.state,
                            "submitted_at": review.submitted_at,
                            "body": review.body
                        }
                        for review in pr_info.reviews
                    ],
                    "status_checks": [
                        {
                            "state": check.state,
                            "context": check.context,
                            "description": check.description
                        }
                        for check in pr_info.status_checks
                    ]
                }
                
                return json.dumps(pr_dict, indent=2)
                
            except Exception as e:
                logger.error(f"Error fetching PR details from GitHub: {e}")
                return json.dumps({
                    "error": str(e),
                    "suggestion": "This might be a network issue or the PR may not exist",
                    "fallback": "Try checking the repository directly on GitHub"
                })
        
        return await self._get_cached_or_fetch(cache_key, _fetch_github_pr)

    async def get_pr_analysis(self, pr_number: Annotated[int, "PR number and description"]) -> str:
        """Get comprehensive PR analysis combining local and GitHub data"""
        cache_key = f"pr_analysis:{pr_number}"
        
        async def _fetch_pr_analysis():
            try:
                # Get local diff data (if available)
                local_data_str = self.get_pr_diff(pr_number)
                local_json = json.loads(local_data_str)
                
                # Get GitHub data  
                github_data_str = await self.get_pr_details_from_github(pr_number)
                github_json = json.loads(github_data_str)
                
                # If local data failed but GitHub data succeeded, use that
                if local_json.get("error") and not github_json.get("error"):
                    local_json = {
                        "note": "Local diff data not available, using GitHub data only",
                        "available": False
                    }
                
                # Combine data
                analysis = {
                    "pr_number": pr_number,
                    "analysis_timestamp": time.time(),
                    "local_data": local_json,
                    "github_data": github_json,
                    "summary": self._generate_pr_summary(pr_number, local_json, github_json)
                }
                
                return json.dumps(analysis, indent=2)
                
            except Exception as e:
                logger.error(f"Error generating PR analysis: {e}")
                return json.dumps({
                    "error": str(e), 
                    "pr_number": pr_number,
                    "suggestion": "Try using get_pr_details_from_github for basic PR information"
                })
        
        return await self._get_cached_or_fetch(cache_key, _fetch_pr_analysis)
    
    def _generate_pr_summary(self, pr_number: int, local_data: Dict[str, Any], github_data: Dict[str, Any]) -> str:
        """Generate a comprehensive PR summary"""
        try:
            summary_parts = []
            
            if github_data and not github_data.get("error"):
                summary_parts.append(f"**PR #{github_data['number']}: {github_data['title']}**")
                summary_parts.append(f"State: {github_data['state']}")
                summary_parts.append(f"Author: {github_data.get('author', 'Unknown')}")
                
                if github_data.get('body'):
                    body_preview = github_data['body'][:200] + "..." if len(github_data['body']) > 200 else github_data['body']
                    summary_parts.append(f"Description: {body_preview}")
                
                # Files changed
                if github_data.get('files_changed'):
                    summary_parts.append(f"Files changed: {len(github_data['files_changed'])} files")
                    summary_parts.append(f"Key files: {', '.join(github_data['files_changed'][:5])}")
                
                # Review status
                if github_data.get('review_decision'):
                    summary_parts.append(f"Review status: {github_data['review_decision']}")
                
                # Stats
                if github_data.get('additions') and github_data.get('deletions'):
                    summary_parts.append(f"Changes: +{github_data['additions']} -{github_data['deletions']}")
            
            elif local_data and not local_data.get("error"):
                summary_parts.append(f"**PR #{pr_number} (from local data)**")
                if local_data.get('diff_summary'):
                    summary_parts.append(f"Summary: {local_data['diff_summary']}")
                if local_data.get('files_changed'):
                    summary_parts.append(f"Files changed: {len(local_data['files_changed'])} files")
            
            return "\n\n".join(summary_parts) if summary_parts else f"PR #{pr_number} - No detailed information available"
            
        except Exception as e:
            return f"PR #{pr_number} - Error generating summary: {str(e)}"
