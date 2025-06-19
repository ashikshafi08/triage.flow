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
        
        # Cache repository context for consistent cache keys
        self._repo_context = None
    
    def _get_repo_context(self) -> str:
        """Get repository context for cache keys"""
        if self._repo_context is None:
            repo_info = self._extract_repo_info(self.repo_path) if self._extract_repo_info else (None, None)
            repo_owner, repo_name = repo_info
            self._repo_context = f"{repo_owner}/{repo_name}" if repo_owner and repo_name else str(self.repo_path)
        return self._repo_context
    
    def _make_cache_key(self, key_type: str, *args) -> str:
        """Create repository-aware cache key"""
        repo_context = self._get_repo_context()
        args_str = ":".join(str(arg) for arg in args)
        return f"{key_type}:{repo_context}:{args_str}" if args_str else f"{key_type}:{repo_context}"

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

    def get_pr_for_issue(self, issue_identifier: Annotated[str, "Issue identifier (number or #number)"]) -> str:
        # Handle different issue identifier formats
        if issue_identifier.startswith('#'):
            issue_number = int(issue_identifier[1:])
        else:
            issue_number = int(issue_identifier)
            
        # Create repository-aware cache key
        cache_key = self._make_cache_key("pr_for_issue", issue_number)
        
        def _fetch_pr_for_issue():
            # First try the issue RAG system if available
            if self.issue_rag_system and hasattr(self.issue_rag_system, 'indexer') and hasattr(self.issue_rag_system.indexer, 'patch_builder'):
                try:
                    patch_builder = self.issue_rag_system.indexer.patch_builder
                    links = patch_builder.load_patch_links().get(issue_number, [])
                    if links:
                        # Convert PatchLink objects to dictionaries
                        pr_data = []
                        for link in links:
                            pr_data.append({
                                "issue_id": link.issue_id,
                                "pr_number": link.pr_number,
                                "merged_at": link.merged_at,
                                "pr_title": link.pr_title,
                                "pr_url": link.pr_url,
                                "pr_diff_url": link.pr_diff_url,
                                "files_changed": link.files_changed
                            })
                        return json.dumps({"issue_number": issue_number, "found_prs": pr_data})
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
        cache_key = self._make_cache_key("pr_diff", pr_number)
        
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
        cache_key = self._make_cache_key("pr_files", pr_number)
        
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
        cache_key = self._make_cache_key("pr_summary", pr_number)
        
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

    def find_open_prs_for_issue(self, issue_number: Annotated[int, "Issue number"]) -> str:
        """Find open pull requests that are related to or reference a specific issue number."""
        try:
            cache_key = f"open_prs_for_issue:{issue_number}"
            
            async def _fetch_open_prs():
                # Try multiple approaches to find open PRs for the issue
                result = {
                    "issue_number": issue_number,
                    "open_prs": [],
                    "search_methods": [],
                    "found_via": None
                }
                
                # Method 1: Use Issue RAG system if available and initialized
                if self.issue_rag_system and self.issue_rag_system.is_initialized():
                    try:
                        # Check for open PR documents in the indexer
                        if hasattr(self.issue_rag_system.indexer, 'open_pr_docs'):
                            open_pr_docs = self.issue_rag_system.indexer.open_pr_docs
                            for pr_number, pr_doc in open_pr_docs.items():
                                # Check if PR mentions this issue
                                pr_text = f"{pr_doc.title} {pr_doc.body}".lower()
                                issue_refs = [f"#{issue_number}", f"issue {issue_number}", f"fixes #{issue_number}", f"closes #{issue_number}"]
                                
                                if any(ref in pr_text for ref in issue_refs):
                                    result["open_prs"].append({
                                        "pr_number": pr_number,
                                        "title": pr_doc.title,
                                        "author": pr_doc.author,
                                        "url": pr_doc.url,
                                        "created_at": pr_doc.created_at,
                                        "updated_at": pr_doc.updated_at,
                                        "review_decision": pr_doc.review_decision,
                                        "draft": pr_doc.draft,
                                        "mergeable": pr_doc.mergeable,
                                        "files_changed": pr_doc.files_changed[:5],  # Limit to first 5 files
                                        "source": "issue_rag_index"
                                    })
                        
                        if result["open_prs"]:
                            result["found_via"] = "issue_rag_index"
                            result["search_methods"].append("issue_rag_index")
                            result["status"] = "found"
                            result["message"] = f"Found {len(result['open_prs'])} open PR(s) related to issue #{issue_number}"
                            return json.dumps(result, indent=2)
                            
                    except Exception as e:
                        logger.warning(f"Issue RAG search failed: {e}")
                        result["search_methods"].append(f"issue_rag_index_failed: {str(e)}")
                
                # Method 2: GitHub API search (if we have access)
                try:
                    from ..github_client import GitHubIssueClient
                    from .utilities import get_repo_url_from_path
                    
                    repo_url = get_repo_url_from_path(self.repo_path)
                    if repo_url:
                        github_client = GitHubIssueClient()
                        
                        # Search for open PRs that mention this issue
                        search_query = f"#{issue_number} type:pr state:open repo:{repo_url.split('github.com/')[-1]}"
                        
                        # Try GitHub search API (may require authentication)
                        try:
                            # This is a simplified search - in practice, you'd use GitHub's search API
                            # For now, we'll use a more direct approach
                            result["search_methods"].append("github_api_attempted")
                            
                        except Exception as api_e:
                            logger.debug(f"GitHub API search failed: {api_e}")
                            result["search_methods"].append(f"github_api_failed: {str(api_e)}")
                            
                except Exception as e:
                    logger.debug(f"GitHub client setup failed: {e}")
                    result["search_methods"].append(f"github_setup_failed: {str(e)}")
                
                # Method 3: Git log search for recent PR references (fallback)
                try:
                    import subprocess
                    import re
                    
                    # Search recent commits for PR merge messages that might reference this issue
                    git_cmd = [
                        "git", "log", "--grep", f"#{issue_number}", 
                        "--pretty=format:%H|%s|%ad", "--date=short", 
                        "--since=30.days.ago", "--all"
                    ]
                    
                    result_proc = subprocess.run(git_cmd, capture_output=True, text=True, cwd=self.repo_path)
                    
                    if result_proc.returncode == 0 and result_proc.stdout.strip():
                        commits_mentioning_issue = []
                        for line in result_proc.stdout.strip().split('\n'):
                            parts = line.split('|', 2)
                            if len(parts) >= 3:
                                commit_sha, subject, date = parts
                                # Look for PR numbers in commit message
                                pr_matches = re.findall(r'(?:pull request|PR|merge.*#|#)(\d+)', subject, re.IGNORECASE)
                                for pr_num in pr_matches:
                                    commits_mentioning_issue.append({
                                        "pr_number": int(pr_num),
                                        "commit_sha": commit_sha[:8],
                                        "subject": subject,
                                        "date": date
                                    })
                        
                        if commits_mentioning_issue:
                            result["recent_commits_mentioning_issue"] = commits_mentioning_issue
                            result["search_methods"].append("git_log_search")
                            # Note: These are likely merged PRs, but still useful context
                    
                except Exception as git_e:
                    logger.debug(f"Git log search failed: {git_e}")
                    result["search_methods"].append(f"git_log_failed: {str(git_e)}")
                
                # Method 4: Search for local branches that might be related PRs
                try:
                    import subprocess
                    import re
                    
                    # List remote branches that might contain the issue number
                    git_cmd = ["git", "branch", "-r", "--format=%(refname:short)"]
                    result_proc = subprocess.run(git_cmd, capture_output=True, text=True, cwd=self.repo_path)
                    
                    if result_proc.returncode == 0:
                        branches = result_proc.stdout.strip().split('\n')
                        related_branches = []
                        
                        for branch in branches:
                            branch = branch.strip()
                            if f"{issue_number}" in branch or f"issue-{issue_number}" in branch or f"fix-{issue_number}" in branch:
                                related_branches.append(branch)
                        
                        if related_branches:
                            result["related_branches"] = related_branches
                            result["search_methods"].append("branch_search")
                    
                except Exception as branch_e:
                    logger.debug(f"Branch search failed: {branch_e}")
                    result["search_methods"].append(f"branch_search_failed: {str(branch_e)}")
                
                # Determine final status
                if result["open_prs"]:
                    result["status"] = "found"
                    result["message"] = f"Found {len(result['open_prs'])} open PR(s) related to issue #{issue_number}"
                else:
                    result["status"] = "not_found"
                    if not result["search_methods"]:
                        result["message"] = f"No search methods available to find open PRs for issue #{issue_number}"
                    else:
                        result["message"] = f"No open PRs found for issue #{issue_number} using methods: {', '.join(result['search_methods'])}"
                
                return json.dumps(result, indent=2)
            
            # Handle async execution properly in sync context
            import asyncio
            import concurrent.futures
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, use thread executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._get_cached_or_fetch(cache_key, _fetch_open_prs)
                    )
                    return future.result(timeout=30)
            else:
                # If no event loop is running, use asyncio.run
                return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_open_prs))
                
        except Exception as e:
            logger.error(f"Error in find_open_prs_for_issue: {e}")
            return json.dumps({
                "error": str(e), 
                "issue_number": issue_number,
                "message": "Failed to search for open PRs. This may be due to system not being fully initialized yet."
            }, indent=2)

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
            try:
                result = {
                    "file_paths": file_paths,
                    "open_prs": [],
                    "search_methods": [],
                    "found_via": None
                }
                
                # Method 1: Use Issue RAG system if available and initialized
                if self.issue_rag_system and self.issue_rag_system.is_initialized():
                    try:
                        # Check for open PR documents in the indexer
                        if hasattr(self.issue_rag_system.indexer, 'open_pr_docs'):
                            open_pr_docs = self.issue_rag_system.indexer.open_pr_docs
                            for pr_number, pr_doc in open_pr_docs.items():
                                # Check if PR touches any of the specified files
                                pr_files = set(pr_doc.files_changed)
                                target_files = set(file_paths)
                                
                                # Check for exact matches or path overlaps
                                matching_files = []
                                for target_file in target_files:
                                    for pr_file in pr_files:
                                        if target_file in pr_file or pr_file in target_file:
                                            matching_files.append(pr_file)
                                
                                if matching_files:
                                    result["open_prs"].append({
                                        "pr_number": pr_number,
                                        "title": pr_doc.title,
                                        "author": pr_doc.author,
                                        "url": pr_doc.url,
                                        "created_at": pr_doc.created_at,
                                        "updated_at": pr_doc.updated_at,
                                        "review_decision": pr_doc.review_decision,
                                        "draft": pr_doc.draft,
                                        "mergeable": pr_doc.mergeable,
                                        "matching_files": matching_files,
                                        "all_files_changed": len(pr_doc.files_changed),
                                        "source": "issue_rag_index"
                                    })
                            
                            if result["open_prs"]:
                                result["found_via"] = "issue_rag_index"
                                result["search_methods"].append("issue_rag_index")
                                result["status"] = "found"
                                result["message"] = f"Found {len(result['open_prs'])} open PR(s) touching the specified files"
                                return json.dumps(result, indent=2)
                                
                    except Exception as e:
                        logger.warning(f"Issue RAG search by files failed: {e}")
                        result["search_methods"].append(f"issue_rag_index_failed: {str(e)}")
                
                # Method 2: Git log search for recent changes to these files
                try:
                    import subprocess
                    import re
                    
                    # Find recent commits that modified these files
                    for file_path in file_paths:
                        git_cmd = [
                            "git", "log", "--pretty=format:%H|%s|%ad", "--date=short",
                            "--since=90.days.ago", "--", file_path
                        ]
                        
                        result_proc = subprocess.run(git_cmd, capture_output=True, text=True, cwd=self.repo_path)
                        
                        if result_proc.returncode == 0 and result_proc.stdout.strip():
                            commits = []
                            for line in result_proc.stdout.strip().split('\n')[:10]:  # Limit to 10 recent commits
                                parts = line.split('|', 2)
                                if len(parts) >= 3:
                                    commit_sha, subject, date = parts
                                    # Look for PR numbers in commit message
                                    pr_matches = re.findall(r'(?:pull request|PR|merge.*#|#)(\d+)', subject, re.IGNORECASE)
                                    if pr_matches:
                                        commits.append({
                                            "file": file_path,
                                            "pr_number": int(pr_matches[0]),
                                            "commit_sha": commit_sha[:8],
                                            "subject": subject,
                                            "date": date
                                        })
                            
                            if commits:
                                if "recent_commits_by_file" not in result:
                                    result["recent_commits_by_file"] = {}
                                result["recent_commits_by_file"][file_path] = commits
                    
                    if "recent_commits_by_file" in result:
                        result["search_methods"].append("git_log_by_file")
                    
                except Exception as git_e:
                    logger.debug(f"Git log search by files failed: {git_e}")
                    result["search_methods"].append(f"git_log_failed: {str(git_e)}")
                
                # Method 3: Check for branches that might be working on these files
                try:
                    import subprocess
                    
                    # Get list of branches
                    git_cmd = ["git", "branch", "-r", "--format=%(refname:short)"]
                    result_proc = subprocess.run(git_cmd, capture_output=True, text=True, cwd=self.repo_path)
                    
                    if result_proc.returncode == 0:
                        branches = result_proc.stdout.strip().split('\n')
                        file_related_branches = []
                        
                        for branch in branches:
                            branch = branch.strip()
                            if branch and not branch.startswith('origin/HEAD'):
                                # Check if branch name contains file-related keywords
                                for file_path in file_paths:
                                    file_name = file_path.split('/')[-1].split('.')[0]  # Get base filename
                                    if len(file_name) > 3 and file_name.lower() in branch.lower():
                                        file_related_branches.append({
                                            "branch": branch,
                                            "related_file": file_path,
                                            "match_reason": f"branch contains filename '{file_name}'"
                                        })
                        
                        if file_related_branches:
                            result["file_related_branches"] = file_related_branches
                            result["search_methods"].append("branch_name_analysis")
                    
                except Exception as branch_e:
                    logger.debug(f"Branch analysis failed: {branch_e}")
                    result["search_methods"].append(f"branch_analysis_failed: {str(branch_e)}")
                
                # Determine final status
                if result["open_prs"]:
                    result["status"] = "found"
                    result["message"] = f"Found {len(result['open_prs'])} open PR(s) touching the specified files"
                else:
                    result["status"] = "not_found"
                    if not result["search_methods"]:
                        result["message"] = f"No search methods available to find open PRs for files: {', '.join(file_paths)}"
                    else:
                        result["message"] = f"No open PRs found touching files {', '.join(file_paths)} using methods: {', '.join(result['search_methods'])}"
                        
                        # Provide helpful context about recent activity
                        if "recent_commits_by_file" in result:
                            recent_pr_numbers = set()
                            for commits in result["recent_commits_by_file"].values():
                                for commit in commits:
                                    recent_pr_numbers.add(commit["pr_number"])
                            if recent_pr_numbers:
                                result["message"] += f". Recent PRs that touched these files: {sorted(recent_pr_numbers)}"
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                logger.error(f"Error searching PRs by files: {e}")
                return json.dumps({
                    "error": str(e),
                    "file_paths": file_paths,
                    "message": "Failed to search for PRs by files. This may be due to system not being fully initialized yet."
                }, indent=2)
        
        if self.pr_cache:
            return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_prs_by_files))
        else:
            return _fetch_prs_by_files()

    def search_open_prs(self, query: Annotated[str, "Search query"], limit: Annotated[int, "Limit"] = 5) -> str:
        """Search through open pull requests by keywords, features, or descriptions to find relevant ones."""
        try:
            cache_key = f"search_prs:{query}:{limit}"
            
            async def _search_prs():
                result = {
                    "query": query,
                    "limit": limit,
                    "open_prs": [],
                    "search_methods": [],
                    "found_via": None
                }
                
                # Method 1: Use Issue RAG system if available and initialized
                if self.issue_rag_system and self.issue_rag_system.is_initialized():
                    try:
                        # Use the issue RAG search functionality to find related open PRs
                        issue_context = await self.issue_rag_system.get_issue_context(
                            query, max_issues=limit, include_patches=True
                        )
                        
                        # Extract open PRs from the patches
                        if hasattr(issue_context, 'patches') and issue_context.patches:
                            for patch in issue_context.patches:
                                if hasattr(patch, 'type') and patch.type == 'open_pr':
                                    result["open_prs"].append({
                                        "pr_number": patch.pr_number,
                                        "title": patch.title,
                                        "author": patch.author,
                                        "url": patch.url,
                                        "similarity": patch.similarity,
                                        "match_reasons": patch.match_reasons,
                                        "source": "issue_rag_search"
                                    })
                        
                        if result["open_prs"]:
                            result["found_via"] = "issue_rag_search"
                            result["search_methods"].append("issue_rag_search")
                            result["status"] = "found"
                            result["message"] = f"Found {len(result['open_prs'])} open PR(s) matching query '{query}'"
                            return json.dumps(result, indent=2)
                            
                    except Exception as e:
                        logger.warning(f"Issue RAG search failed: {e}")
                        result["search_methods"].append(f"issue_rag_search_failed: {str(e)}")
                
                # Method 2: Search open PR documents directly if available
                if self.issue_rag_system and hasattr(self.issue_rag_system, 'indexer') and hasattr(self.issue_rag_system.indexer, 'open_pr_docs'):
                    try:
                        open_pr_docs = self.issue_rag_system.indexer.open_pr_docs
                        query_lower = query.lower()
                        
                        for pr_number, pr_doc in open_pr_docs.items():
                            # Simple text matching in title and body
                            pr_text = f"{pr_doc.title} {pr_doc.body}".lower()
                            
                            # Calculate basic relevance score
                            query_words = query_lower.split()
                            matches = sum(1 for word in query_words if word in pr_text)
                            relevance_score = matches / len(query_words) if query_words else 0
                            
                            if relevance_score > 0.3:  # At least 30% of query words match
                                result["open_prs"].append({
                                    "pr_number": pr_number,
                                    "title": pr_doc.title,
                                    "author": pr_doc.author,
                                    "url": pr_doc.url,
                                    "created_at": pr_doc.created_at,
                                    "updated_at": pr_doc.updated_at,
                                    "review_decision": pr_doc.review_decision,
                                    "draft": pr_doc.draft,
                                    "relevance_score": relevance_score,
                                    "source": "direct_text_search"
                                })
                        
                        # Sort by relevance score
                        result["open_prs"].sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                        result["open_prs"] = result["open_prs"][:limit]
                        
                        if result["open_prs"]:
                            result["found_via"] = "direct_text_search"
                            result["search_methods"].append("direct_text_search")
                            result["status"] = "found"
                            result["message"] = f"Found {len(result['open_prs'])} open PR(s) matching query '{query}'"
                            return json.dumps(result, indent=2)
                            
                    except Exception as e:
                        logger.warning(f"Direct text search failed: {e}")
                        result["search_methods"].append(f"direct_text_search_failed: {str(e)}")
                
                # Method 3: Git log search for branches/PRs mentioning the query terms
                try:
                    import subprocess
                    import re
                    
                    # Search commit messages for the query terms
                    git_cmd = [
                        "git", "log", "--grep", query, "--pretty=format:%H|%s|%ad",
                        "--date=short", "--since=90.days.ago", "--all"
                    ]
                    
                    result_proc = subprocess.run(git_cmd, capture_output=True, text=True, cwd=self.repo_path)
                    
                    if result_proc.returncode == 0 and result_proc.stdout.strip():
                        relevant_commits = []
                        for line in result_proc.stdout.strip().split('\n')[:limit*2]:  # Get more commits to find PRs
                            parts = line.split('|', 2)
                            if len(parts) >= 3:
                                commit_sha, subject, date = parts
                                # Look for PR numbers in commit message
                                pr_matches = re.findall(r'(?:pull request|PR|merge.*#|#)(\d+)', subject, re.IGNORECASE)
                                if pr_matches:
                                    relevant_commits.append({
                                        "pr_number": int(pr_matches[0]),
                                        "commit_sha": commit_sha[:8],
                                        "subject": subject,
                                        "date": date
                                    })
                        
                        if relevant_commits:
                            result["relevant_commits"] = relevant_commits
                            result["search_methods"].append("git_log_search")
                    
                except Exception as git_e:
                    logger.debug(f"Git log search failed: {git_e}")
                    result["search_methods"].append(f"git_log_failed: {str(git_e)}")
                
                # Determine final status
                if result["open_prs"]:
                    result["status"] = "found"
                    result["message"] = f"Found {len(result['open_prs'])} open PR(s) matching query '{query}'"
                else:
                    result["status"] = "not_found"
                    if not result["search_methods"]:
                        result["message"] = f"No search methods available to find open PRs for query '{query}'"
                    else:
                        result["message"] = f"No open PRs found matching query '{query}' using methods: {', '.join(result['search_methods'])}"
                
                return json.dumps(result, indent=2)
            
            # Handle async execution properly in sync context
            import asyncio
            import concurrent.futures
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, use thread executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._get_cached_or_fetch(cache_key, _search_prs)
                    )
                    return future.result(timeout=30)
            else:
                # If no event loop is running, use asyncio.run
                return asyncio.run(self._get_cached_or_fetch(cache_key, _search_prs))
                
        except Exception as e:
            logger.error(f"Error in search_open_prs: {e}")
            return json.dumps({"error": str(e), "query": query})

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

    def get_pr_details_from_github(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Get comprehensive PR details from GitHub API with caching"""
        try:
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
            
            # Handle async execution properly in sync context
            import asyncio
            import concurrent.futures
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, use thread executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._get_cached_or_fetch(cache_key, _fetch_github_pr)
                    )
                    return future.result(timeout=30)
            else:
                # If no event loop is running, use asyncio.run
                return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_github_pr))
                
        except Exception as e:
            logger.error(f"Error in get_pr_details_from_github: {e}")
            return json.dumps({"error": str(e), "pr_number": pr_number})

    def get_pr_analysis(self, pr_number: Annotated[int, "PR number and description"]) -> str:
        """Get comprehensive PR analysis combining local and GitHub data"""
        try:
            cache_key = f"pr_analysis:{pr_number}"
            
            async def _fetch_pr_analysis():
                try:
                    # Get local diff data (if available)
                    local_data_str = self.get_pr_diff(pr_number)
                    local_json = json.loads(local_data_str)
                    
                    # Get GitHub data - note: this is now a sync function
                    github_data_str = self.get_pr_details_from_github(pr_number)
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
            
            # Handle async execution properly in sync context
            import asyncio
            import concurrent.futures
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, use thread executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._get_cached_or_fetch(cache_key, _fetch_pr_analysis)
                    )
                    return future.result(timeout=30)
            else:
                # If no event loop is running, use asyncio.run
                return asyncio.run(self._get_cached_or_fetch(cache_key, _fetch_pr_analysis))
                
        except Exception as e:
            logger.error(f"Error in get_pr_analysis: {e}")
            return json.dumps({"error": str(e), "pr_number": pr_number})
    
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
